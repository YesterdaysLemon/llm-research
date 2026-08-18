from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
P0_PATH = Path(__file__).with_name("run_pilot.py")
P0_SPEC = importlib.util.spec_from_file_location("e001_run_pilot", P0_PATH)
assert P0_SPEC is not None and P0_SPEC.loader is not None
p0 = importlib.util.module_from_spec(P0_SPEC)
sys.modules[P0_SPEC.name] = p0
P0_SPEC.loader.exec_module(p0)


def make_task_table(config: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    family = str(config["task_family"])
    num_entities = int(config["data"]["num_entities"])
    num_relations = int(config["data"]["num_relations"])
    seed = int(config["data_seed"])
    if family == "affine":
        return p0.make_relation_table(num_entities, num_relations, seed)
    if family != "bitwise":
        raise ValueError(f"unsupported task family: {family}")
    if num_entities < 2 or num_entities & (num_entities - 1):
        raise ValueError("bitwise task requires a power-of-two entity count")

    bits = int(math.log2(num_entities))
    rng = np.random.default_rng(seed)
    masks = rng.choice(num_entities, size=num_relations, replace=False)
    entities = np.arange(num_entities, dtype=np.int64)
    table = np.empty((num_relations, num_entities), dtype=np.int64)
    definitions: list[dict[str, Any]] = []
    low_mask = num_entities - 1
    for relation, mask in enumerate(masks):
        shift = relation % bits
        if shift:
            rotated = ((entities << shift) | (entities >> (bits - shift))) & low_mask
        else:
            rotated = entities.copy()
        table[relation] = np.bitwise_xor(rotated, int(mask))
        definitions.append(
            {
                "relation": relation,
                "operation": "rotate_left_then_xor",
                "shift": shift,
                "xor_mask": int(mask),
            }
        )
    return table, definitions


def tensor_bytes(tensor: Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def dataset_bytes(dataset: Any) -> int:
    return sum(
        tensor_bytes(tensor)
        for tensor in (dataset.tokens, dataset.labels, dataset.depths)
    )


def self_relational_loss(student: Tensor) -> Tensor:
    if student.shape[1] < 2:
        raise ValueError("self-relation control requires at least two student layers")
    target = p0.normalized_gram(student[:, 0].detach())
    losses = [
        (p0.normalized_gram(student[:, layer]) - target).pow(2).sum()
        for layer in range(1, student.shape[1])
    ]
    return torch.stack(losses).mean()


def train_student(
    model: nn.Module,
    dataset: Any,
    *,
    condition: str,
    teacher_logits: Tensor,
    teacher_trajectory: Tensor,
    untrained_trajectory: Tensor,
    random_trajectory: Tensor,
    fixed_shuffled_trajectory: Tensor,
    epochs: int,
    auxiliary_weight: float,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    p0.set_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    loader = p0.make_loader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(training_config["num_workers"]),
    )
    start = time.perf_counter()
    last_loss = float("nan")
    last_base_loss = float("nan")
    last_auxiliary_loss = 0.0
    steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    label_conditions = {"labels", "labels_rich", "labels_time"}
    for _ in range(epochs):
        for indices, tokens, labels, _ in loader:
            indices = indices.to(torch.long)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            student_logits, student_trajectory = model(tokens)

            if condition in label_conditions:
                base_loss = F.cross_entropy(student_logits, labels)
            else:
                batch_teacher_logits = teacher_logits[indices].to(
                    device, non_blocking=True
                )
                base_loss = p0.distillation_loss(
                    student_logits,
                    batch_teacher_logits,
                    labels,
                    temperature=float(training_config["temperature"]),
                    label_weight=float(training_config["label_weight"]),
                    logit_weight=float(training_config["logit_weight"]),
                )

            auxiliary = torch.zeros((), device=device)
            if condition == "pointwise":
                target = teacher_trajectory[indices].to(device, non_blocking=True)
                if target.shape[-1] != student_trajectory.shape[-1]:
                    raise ValueError("pointwise condition requires equal hidden widths")
                auxiliary = p0.pointwise_loss(student_trajectory, target)
            elif condition in {
                "relational",
                "shuffled",
                "fixed_shuffled",
                "random",
                "untrained",
            }:
                target_cache = {
                    "relational": teacher_trajectory,
                    "shuffled": teacher_trajectory,
                    "fixed_shuffled": fixed_shuffled_trajectory,
                    "random": random_trajectory,
                    "untrained": untrained_trajectory,
                }[condition]
                target = target_cache[indices].to(device, non_blocking=True)
                if condition == "shuffled":
                    target = target[torch.randperm(target.shape[0], device=device)]
                auxiliary = p0.relational_loss(student_trajectory, target)
            elif condition == "self":
                auxiliary = self_relational_loss(student_trajectory)

            loss = base_loss + auxiliary_weight * auxiliary
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config["gradient_clip"])
            )
            optimizer.step()
            steps += 1
            last_loss = float(loss.detach().item())
            last_base_loss = float(base_loss.detach().item())
            last_auxiliary_loss = float(auxiliary.detach().item())

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())
    else:
        peak_memory = 0
    return {
        "train_seconds": time.perf_counter() - start,
        "optimizer_steps": steps,
        "example_exposures": len(dataset) * epochs,
        "last_batch_loss": last_loss,
        "last_base_loss": last_base_loss,
        "last_auxiliary_loss": last_auxiliary_loss,
        "peak_gpu_memory_bytes": peak_memory,
    }


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["trial_id"])].append(run)
    summary: list[dict[str, Any]] = []
    for trial_id, rows in grouped.items():
        row: dict[str, Any] = {
            "trial_id": trial_id,
            "condition": rows[0]["condition"],
            "epochs": rows[0]["epochs"],
            "auxiliary_weight": rows[0]["auxiliary_weight"],
            "runs": len(rows),
        }
        for metric in (
            "id_accuracy",
            "evaluation_accuracy",
            "id_cross_entropy",
            "evaluation_cross_entropy",
            "train_seconds",
            "peak_gpu_memory_bytes",
        ):
            values = np.asarray([float(item[metric]) for item in rows])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary.append(row)
    return sorted(summary, key=lambda item: str(item["trial_id"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("high")

    data = config["data"]
    training = config["training"]
    table, relation_definitions = make_task_table(config)
    all_depths = (
        list(data["teacher_depths"])
        + list(data["student_depths"])
        + list(data["id_depths"])
        + list(data["evaluation_depths"])
    )
    max_depth = max(all_depths)
    excluded_pairs = list(data["excluded_pairs"])
    evaluation_pairs = list(data["evaluation_pairs"])
    evaluation_forbidden_pairs = list(data.get("evaluation_forbidden_pairs", []))

    teacher_train = p0.CompositionDataset(
        count=int(data["teacher_train_examples"]),
        depths=list(data["teacher_depths"]),
        table=table,
        seed=int(config["data_seed"]) + 1,
        max_depth=max_depth,
    )
    student_train = p0.CompositionDataset(
        count=int(data["student_train_examples"]),
        depths=list(data["student_depths"]),
        table=table,
        seed=int(config["data_seed"]) + 2,
        max_depth=max_depth,
        forbidden_pairs=excluded_pairs,
    )
    rich_train = p0.CompositionDataset(
        count=int(data["rich_train_examples"]),
        depths=list(data["student_depths"]),
        table=table,
        seed=int(config["data_seed"]) + 20,
        max_depth=max_depth,
        forbidden_pairs=excluded_pairs,
    )
    id_test = p0.CompositionDataset(
        count=int(data["id_test_examples"]),
        depths=list(data["id_depths"]),
        table=table,
        seed=int(config["data_seed"]) + 3,
        max_depth=max_depth,
        forbidden_pairs=excluded_pairs,
    )
    evaluation_test = p0.CompositionDataset(
        count=int(data["evaluation_examples"]),
        depths=list(data["evaluation_depths"]),
        table=table,
        seed=int(config["data_seed"]) + int(data["evaluation_seed_offset"]),
        max_depth=max_depth,
        forbidden_pairs=evaluation_forbidden_pairs,
        required_pairs=evaluation_pairs,
    )

    p0.set_seed(int(config["teacher_seed"]))
    teacher = p0.model_from_config(config["teacher"], teacher_train).to(device)
    teacher_training = p0.train_teacher(
        teacher,
        teacher_train,
        config=config["teacher"],
        training_config=training,
        seed=int(config["teacher_seed"]),
        device=device,
    )
    teacher_id = p0.evaluate(
        teacher,
        id_test,
        device=device,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
    )
    teacher_evaluation = p0.evaluate(
        teacher,
        evaluation_test,
        device=device,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
    )

    student_layers = int(config["student"]["n_layers"])
    teacher_logits, teacher_trajectory, selected_layers = p0.cache_teacher(
        teacher,
        student_train,
        student_layers=student_layers,
        device=device,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
    )
    p0.set_seed(int(config["feature_teacher_seed"]))
    untrained_teacher = p0.model_from_config(config["teacher"], teacher_train).to(device)
    _, untrained_trajectory, untrained_layers = p0.cache_teacher(
        untrained_teacher,
        student_train,
        student_layers=student_layers,
        device=device,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
    )
    del untrained_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    target_generator = torch.Generator().manual_seed(int(config["target_seed"]))
    random_trajectory = torch.randn(
        teacher_trajectory.shape,
        generator=target_generator,
        dtype=teacher_trajectory.dtype,
    )
    permutation = torch.randperm(
        len(student_train), generator=target_generator
    )
    fixed_shuffled_trajectory = teacher_trajectory[permutation]

    runs: list[dict[str, Any]] = []
    student_parameters = 0
    for seed in config["student_seeds"]:
        for trial in config["trials"]:
            condition = str(trial["condition"])
            trial_id = str(trial["id"])
            epochs = int(trial["epochs"])
            auxiliary_weight = float(trial.get("auxiliary_weight", 0.0))
            active_dataset = rich_train if condition == "labels_rich" else student_train
            p0.set_seed(int(seed))
            student = p0.model_from_config(config["student"], active_dataset).to(device)
            student_parameters = p0.count_parameters(student)
            train_metrics = train_student(
                student,
                active_dataset,
                condition=condition,
                teacher_logits=teacher_logits,
                teacher_trajectory=teacher_trajectory,
                untrained_trajectory=untrained_trajectory,
                random_trajectory=random_trajectory,
                fixed_shuffled_trajectory=fixed_shuffled_trajectory,
                epochs=epochs,
                auxiliary_weight=auxiliary_weight,
                model_config=config["student"],
                training_config=training,
                seed=int(seed),
                device=device,
            )
            id_metrics = p0.evaluate(
                student,
                id_test,
                device=device,
                batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]),
            )
            evaluation_metrics = p0.evaluate(
                student,
                evaluation_test,
                device=device,
                batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]),
            )
            row = {
                "trial_id": trial_id,
                "condition": condition,
                "seed": int(seed),
                "epochs": epochs,
                "auxiliary_weight": auxiliary_weight,
                "training_examples": len(active_dataset),
                "training_dataset_bytes": dataset_bytes(active_dataset),
                "id_accuracy": id_metrics.accuracy,
                "id_cross_entropy": id_metrics.cross_entropy,
                "id_accuracy_by_depth": id_metrics.accuracy_by_depth,
                "evaluation_accuracy": evaluation_metrics.accuracy,
                "evaluation_cross_entropy": evaluation_metrics.cross_entropy,
                "evaluation_accuracy_by_depth": evaluation_metrics.accuracy_by_depth,
                **train_metrics,
            }
            runs.append(row)
            print(
                json.dumps(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "id_accuracy": id_metrics.accuracy,
                        "evaluation_accuracy": evaluation_metrics.accuracy,
                        "train_seconds": train_metrics["train_seconds"],
                    }
                ),
                flush=True,
            )
            del student
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize(runs)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": config["experiment_id"],
        "phase": config["phase"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": p0.sha256_file(config_path),
        "git_commit": p0.git_value("rev-parse", "HEAD"),
        "git_status_porcelain": p0.git_value("status", "--porcelain=v1"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
            ),
            "gpu_memory_bytes": (
                int(torch.cuda.get_device_properties(0).total_memory)
                if device.type == "cuda"
                else 0
            ),
            "gpu_power_draw_w": p0.nvidia_value("power.draw"),
        },
        "task_family": config["task_family"],
        "relation_definitions": relation_definitions,
        "teacher": {
            "parameters": p0.count_parameters(teacher),
            "selected_trajectory_layers_zero_based": selected_layers,
            "untrained_selected_layers_zero_based": untrained_layers,
            "training": teacher_training,
            "id": p0.asdict(teacher_id),
            "evaluation": p0.asdict(teacher_evaluation),
        },
        "student_parameters": student_parameters,
        "byte_accounting": {
            "standard_training_dataset": dataset_bytes(student_train),
            "rich_training_dataset": dataset_bytes(rich_train),
            "teacher_logits": tensor_bytes(teacher_logits),
            "teacher_trajectory": tensor_bytes(teacher_trajectory),
            "untrained_trajectory": tensor_bytes(untrained_trajectory),
            "random_trajectory": tensor_bytes(random_trajectory),
        },
        "runs": runs,
        "summary": summary,
        "limitations": [
            "Synthetic composition task, not natural language.",
            "In-memory tensor bytes are not an information-theoretic minimum.",
            "Energy is unavailable if GPU power telemetry reports N/A.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), summary)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
