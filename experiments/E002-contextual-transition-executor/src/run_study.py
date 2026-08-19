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


ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT_PATH = Path(__file__).with_name("run_development.py")
DEVELOPMENT_SPEC = importlib.util.spec_from_file_location(
    "e002_development", DEVELOPMENT_PATH
)
assert DEVELOPMENT_SPEC is not None and DEVELOPMENT_SPEC.loader is not None
development = importlib.util.module_from_spec(DEVELOPMENT_SPEC)
sys.modules[DEVELOPMENT_SPEC.name] = development
DEVELOPMENT_SPEC.loader.exec_module(development)
p0 = development.p0


def make_task_table(config: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data = config["data"]
    family = str(config["task_family"])
    num_entities = int(data["num_entities"])
    num_relations = int(data["num_relations"])
    seed = int(config["data_seed"])
    if family == "affine":
        return p0.make_relation_table(num_entities, num_relations, seed)
    if family != "bitwise":
        raise ValueError(f"unsupported task family: {family}")
    if num_entities < 2 or num_entities & (num_entities - 1):
        raise ValueError("bitwise task requires a power-of-two entity count")
    bits = int(math.log2(num_entities))
    low_mask = num_entities - 1
    rng = np.random.default_rng(seed)
    masks = rng.choice(num_entities, size=num_relations, replace=False)
    entities = np.arange(num_entities, dtype=np.int64)
    table = np.empty((num_relations, num_entities), dtype=np.int64)
    definitions: list[dict[str, Any]] = []
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


def trajectory_steps(spec: dict[str, Any], max_depth: int) -> int:
    if spec["type"] == "transformer":
        return int(spec["n_layers"])
    return max_depth


def label_geometry_targets(
    labels: Tensor, *, num_classes: int, steps: int
) -> Tensor:
    """Build a teacher-free, layer-indexed target from terminal-label identity."""
    one_hot = F.one_hot(labels.to(torch.long), num_classes=num_classes).to(
        torch.float32
    )
    return one_hot.unsqueeze(1).expand(-1, steps, -1).contiguous()


def train_model(
    model: nn.Module,
    dataset: Any,
    *,
    spec: dict[str, Any],
    teacher_logits: Tensor,
    teacher_trajectory: Tensor,
    label_trajectory: Tensor,
    untrained_trajectory: Tensor,
    random_trajectory: Tensor,
    seed: int,
    training: dict[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    p0.set_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loader = p0.make_loader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(training["num_workers"]),
    )
    condition = str(spec["condition"])
    auxiliary_weight = float(spec.get("auxiliary_weight", 0.0))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    steps = 0
    last_loss = float("nan")
    last_base_loss = float("nan")
    last_auxiliary_loss = 0.0
    for _ in range(int(spec["epochs"])):
        for indices, tokens, labels, _ in loader:
            indices = indices.to(torch.long)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, trajectory = model(tokens)
            if condition == "labels":
                base_loss = F.cross_entropy(logits, labels)
            else:
                target_logits = teacher_logits[indices].to(device, non_blocking=True)
                base_loss = p0.distillation_loss(
                    logits,
                    target_logits,
                    labels,
                    temperature=float(training["temperature"]),
                    label_weight=float(training["label_weight"]),
                    logit_weight=float(training["logit_weight"]),
                )
            auxiliary = torch.zeros((), device=device)
            if condition in {
                "relational",
                "label_geometry",
                "shuffled",
                "random",
                "untrained",
            }:
                target_cache = {
                    "relational": teacher_trajectory,
                    "label_geometry": label_trajectory,
                    "shuffled": teacher_trajectory,
                    "random": random_trajectory,
                    "untrained": untrained_trajectory,
                }[condition]
                target = target_cache[indices].to(device, non_blocking=True)
                if condition == "shuffled":
                    target = target[torch.randperm(target.shape[0], device=device)]
                auxiliary = p0.relational_loss(trajectory, target)
            loss = base_loss + auxiliary_weight * auxiliary
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
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
        "example_exposures": len(dataset) * int(spec["epochs"]),
        "last_batch_loss": last_loss,
        "last_base_loss": last_base_loss,
        "last_auxiliary_loss": last_auxiliary_loss,
        "peak_gpu_memory_bytes": peak_memory,
    }


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["condition_id"])].append(run)
    summaries: list[dict[str, Any]] = []
    for condition_id, rows in grouped.items():
        item: dict[str, Any] = {
            "condition_id": condition_id,
            "model_type": rows[0]["model_type"],
            "condition": rows[0]["condition"],
            "stored_parameters": rows[0]["stored_parameters"],
            "active_parameters_per_step": rows[0]["active_parameters_per_step"],
            "runs": len(rows),
        }
        for metric in (
            "id_accuracy",
            "evaluation_accuracy",
            "train_seconds",
            "peak_gpu_memory_bytes",
        ):
            values = np.asarray([float(row[metric]) for row in rows])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summaries.append(item)
    return sorted(summaries, key=lambda item: str(item["condition_id"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    table, definitions = make_task_table(config)
    max_depth = max(
        list(data["teacher_depths"])
        + list(data["train_depths"])
        + list(data["id_depths"])
        + list(data["evaluation_depths"])
    )
    train_data = p0.CompositionDataset(
        count=int(data["train_examples"]),
        depths=list(data["train_depths"]),
        table=table,
        seed=int(config["data_seed"]) + 1,
        max_depth=max_depth,
        forbidden_pairs=list(data["excluded_pairs"]),
    )
    teacher_data = p0.CompositionDataset(
        count=int(data["teacher_train_examples"]),
        depths=list(data["teacher_depths"]),
        table=table,
        seed=int(config["data_seed"]) + 10,
        max_depth=max_depth,
    )
    id_data = p0.CompositionDataset(
        count=int(data["id_examples"]),
        depths=list(data["id_depths"]),
        table=table,
        seed=int(config["data_seed"]) + 2,
        max_depth=max_depth,
        forbidden_pairs=list(data["excluded_pairs"]),
    )
    evaluation_data = p0.CompositionDataset(
        count=int(data["evaluation_examples"]),
        depths=list(data["evaluation_depths"]),
        table=table,
        seed=int(config["data_seed"]) + int(data["evaluation_seed_offset"]),
        max_depth=max_depth,
        forbidden_pairs=list(data["evaluation_forbidden_pairs"]),
        required_pairs=list(data["evaluation_pairs"]),
    )

    p0.set_seed(int(config["teacher_seed"]))
    teacher = p0.model_from_config(config["teacher"], teacher_data).to(device)
    teacher_training = p0.train_teacher(
        teacher,
        teacher_data,
        config=config["teacher"],
        training_config=training,
        seed=int(config["teacher_seed"]),
        device=device,
    )
    teacher_id = p0.evaluate(
        teacher,
        id_data,
        device=device,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
    )
    teacher_evaluation = p0.evaluate(
        teacher,
        evaluation_data,
        device=device,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
    )

    required_steps = sorted(
        {trajectory_steps(spec, max_depth) for spec in config["conditions"]}
    )
    target_caches: dict[int, dict[str, Tensor]] = {}
    for steps in required_steps:
        logits_cache, trajectory_cache, _ = p0.cache_teacher(
            teacher,
            train_data,
            student_layers=steps,
            device=device,
            batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]),
        )
        p0.set_seed(int(config["feature_teacher_seed"]))
        untrained_teacher = p0.model_from_config(config["teacher"], teacher_data).to(device)
        _, untrained_cache, _ = p0.cache_teacher(
            untrained_teacher,
            train_data,
            student_layers=steps,
            device=device,
            batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]),
        )
        del untrained_teacher
        generator = torch.Generator().manual_seed(int(config["target_seed"]) + steps)
        random_cache = torch.randn(
            trajectory_cache.shape,
            generator=generator,
            dtype=trajectory_cache.dtype,
        )
        target_caches[steps] = {
            "logits": logits_cache,
            "trajectory": trajectory_cache,
            "label_trajectory": label_geometry_targets(
                train_data.labels,
                num_classes=int(data["num_entities"]),
                steps=steps,
            ),
            "untrained": untrained_cache,
            "random": random_cache,
        }
    if device.type == "cuda":
        torch.cuda.empty_cache()

    runs: list[dict[str, Any]] = []
    for seed in config["student_seeds"]:
        for spec in config["conditions"]:
            p0.set_seed(int(seed))
            model = development.build_model(spec, train_data).to(device)
            stored = p0.count_parameters(model)
            active = development.active_parameters(model)
            steps = trajectory_steps(spec, max_depth)
            caches = target_caches[steps]
            training_metrics = train_model(
                model,
                train_data,
                spec=spec,
                teacher_logits=caches["logits"],
                teacher_trajectory=caches["trajectory"],
                label_trajectory=caches["label_trajectory"],
                untrained_trajectory=caches["untrained"],
                random_trajectory=caches["random"],
                seed=int(seed),
                training=training,
                device=device,
            )
            id_metrics = p0.evaluate(
                model,
                id_data,
                device=device,
                batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]),
            )
            evaluation_metrics = p0.evaluate(
                model,
                evaluation_data,
                device=device,
                batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]),
            )
            row = {
                "condition_id": spec["id"],
                "model_type": spec["type"],
                "condition": spec["condition"],
                "seed": int(seed),
                "stored_parameters": stored,
                "active_parameters_per_step": active,
                "epochs": int(spec["epochs"]),
                "auxiliary_weight": float(spec.get("auxiliary_weight", 0.0)),
                "id_accuracy": id_metrics.accuracy,
                "id_cross_entropy": id_metrics.cross_entropy,
                "id_accuracy_by_depth": id_metrics.accuracy_by_depth,
                "evaluation_accuracy": evaluation_metrics.accuracy,
                "evaluation_cross_entropy": evaluation_metrics.cross_entropy,
                "evaluation_accuracy_by_depth": evaluation_metrics.accuracy_by_depth,
                **training_metrics,
            }
            runs.append(row)
            print(
                json.dumps(
                    {
                        "condition_id": spec["id"],
                        "seed": seed,
                        "stored_parameters": stored,
                        "evaluation_accuracy": evaluation_metrics.accuracy,
                        "evaluation_by_depth": evaluation_metrics.accuracy_by_depth,
                        "train_seconds": training_metrics["train_seconds"],
                    }
                ),
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize(runs)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": config["experiment_id"],
        "phase": config["phase"],
        "task_family": config["task_family"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": p0.sha256_file(config_path),
        "git_commit": p0.git_value("rev-parse", "HEAD"),
        "git_status_porcelain": p0.git_value("status", "--porcelain=v1"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
            ),
            "gpu_power_draw_w": p0.nvidia_value("power.draw"),
        },
        "relation_definitions": definitions,
        "teacher": {
            "parameters": p0.count_parameters(teacher),
            "training": teacher_training,
            "id": p0.asdict(teacher_id),
            "evaluation": p0.asdict(teacher_evaluation),
        },
        "runs": runs,
        "summary": summary,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), summary)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
