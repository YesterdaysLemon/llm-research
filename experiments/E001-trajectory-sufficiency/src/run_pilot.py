from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def nvidia_value(field: str) -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def make_relation_table(
    num_entities: int, num_relations: int, seed: int
) -> tuple[np.ndarray, list[dict[str, int]]]:
    if num_entities < 3:
        raise ValueError("num_entities must be at least 3")
    rng = np.random.default_rng(seed)
    valid_a = [a for a in range(1, num_entities) if math.gcd(a, num_entities) == 1]
    if len(valid_a) < num_relations:
        raise ValueError("not enough invertible affine multipliers")

    multipliers = rng.choice(valid_a, size=num_relations, replace=False)
    offsets = rng.choice(num_entities, size=num_relations, replace=False)
    entities = np.arange(num_entities, dtype=np.int64)
    table = np.empty((num_relations, num_entities), dtype=np.int64)
    definitions: list[dict[str, int]] = []
    for relation, (a, b) in enumerate(zip(multipliers, offsets, strict=True)):
        table[relation] = (int(a) * entities + int(b)) % num_entities
        definitions.append({"relation": relation, "a": int(a), "b": int(b)})
    return table, definitions


def apply_relations(start: int, relations: Iterable[int], table: np.ndarray) -> int:
    state = int(start)
    for relation in relations:
        state = int(table[int(relation), state])
    return state


class CompositionDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        *,
        count: int,
        depths: list[int],
        table: np.ndarray,
        seed: int,
        max_depth: int,
        forbidden_pairs: list[list[int]] | None = None,
        required_pairs: list[list[int]] | None = None,
    ) -> None:
        self.num_entities = int(table.shape[1])
        self.num_relations = int(table.shape[0])
        self.cls_id = self.num_entities + self.num_relations
        self.pad_id = self.cls_id + 1
        self.max_length = max_depth + 2

        rng = np.random.default_rng(seed)
        tokens = np.full((count, self.max_length), self.pad_id, dtype=np.int64)
        labels = np.empty(count, dtype=np.int64)
        sample_depths = np.empty(count, dtype=np.int64)

        forbidden = {tuple(pair) for pair in (forbidden_pairs or [])}
        required = {tuple(pair) for pair in (required_pairs or [])}
        index = 0
        attempts = 0
        max_attempts = max(10_000, count * 1_000)
        while index < count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("could not generate enough constrained examples")
            depth = int(rng.choice(depths))
            start = int(rng.integers(self.num_entities))
            relations = rng.integers(self.num_relations, size=depth, dtype=np.int64)
            adjacent_pairs = {
                (int(left), int(right))
                for left, right in zip(relations[:-1], relations[1:], strict=True)
            }
            if forbidden and adjacent_pairs.intersection(forbidden):
                continue
            if required and not adjacent_pairs.intersection(required):
                continue
            sequence = [self.cls_id, start]
            sequence.extend((relations + self.num_entities).tolist())
            tokens[index, : len(sequence)] = sequence
            labels[index] = apply_relations(start, relations, table)
            sample_depths[index] = depth
            index += 1

        self.tokens = torch.from_numpy(tokens)
        self.labels = torch.from_numpy(labels)
        self.depths = torch.from_numpy(sample_depths)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            torch.tensor(index, dtype=torch.long),
            self.tokens[index],
            self.labels[index],
            self.depths[index],
        )


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        max_length: int,
        pad_id: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).expand(batch, length)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        padding_mask = tokens.eq(self.pad_id)
        trajectory: list[Tensor] = []
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
            trajectory.append(self.final_norm(hidden[:, 0]))
        pooled = self.final_norm(hidden[:, 0])
        return self.classifier(pooled), torch.stack(trajectory, dim=1)


def model_from_config(
    config: dict[str, Any],
    dataset: CompositionDataset,
) -> TrajectoryTransformer:
    return TrajectoryTransformer(
        vocab_size=dataset.pad_id + 1,
        num_classes=dataset.num_entities,
        max_length=dataset.max_length,
        pad_id=dataset.pad_id,
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        d_ff=int(config["d_ff"]),
        dropout=float(config["dropout"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@dataclass
class EvalMetrics:
    accuracy: float
    cross_entropy: float
    count: int
    accuracy_by_depth: dict[str, float]


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: CompositionDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> EvalMetrics:
    model.eval()
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
    )
    correct = 0
    total = 0
    loss_sum = 0.0
    depth_correct: defaultdict[int, int] = defaultdict(int)
    depth_total: defaultdict[int, int] = defaultdict(int)

    for _, tokens, labels, depths in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = model(tokens)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        predictions = logits.argmax(dim=-1)
        matches = predictions.eq(labels).cpu()
        correct += int(matches.sum().item())
        total += int(labels.numel())
        for depth, match in zip(depths.tolist(), matches.tolist(), strict=True):
            depth_total[int(depth)] += 1
            depth_correct[int(depth)] += int(match)

    return EvalMetrics(
        accuracy=correct / total,
        cross_entropy=loss_sum / total,
        count=total,
        accuracy_by_depth={
            str(depth): depth_correct[depth] / depth_total[depth]
            for depth in sorted(depth_total)
        },
    )


def train_teacher(
    model: nn.Module,
    dataset: CompositionDataset,
    *,
    config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    loader = make_loader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(training_config["num_workers"]),
    )
    start = time.perf_counter()
    last_loss = float("nan")
    for _ in range(int(config["epochs"])):
        for _, tokens, labels, _ in loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tokens)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config["gradient_clip"])
            )
            optimizer.step()
            last_loss = float(loss.detach().item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "train_seconds": time.perf_counter() - start,
        "last_batch_loss": last_loss,
    }


@torch.inference_mode()
def cache_teacher(
    model: nn.Module,
    dataset: CompositionDataset,
    *,
    student_layers: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[Tensor, Tensor, list[int]]:
    model.eval()
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
    )
    logits_cache: list[Tensor] = []
    trajectory_cache: list[Tensor] = []
    teacher_layers = len(model.layers)  # type: ignore[attr-defined]
    selected = (
        torch.linspace(0, teacher_layers - 1, steps=student_layers)
        .round()
        .to(torch.long)
        .tolist()
    )
    for _, tokens, _, _ in loader:
        tokens = tokens.to(device, non_blocking=True)
        logits, trajectory = model(tokens)
        logits_cache.append(logits.cpu())
        trajectory_cache.append(trajectory[:, selected].cpu())
    return torch.cat(logits_cache), torch.cat(trajectory_cache), selected


def normalized_gram(features: Tensor, epsilon: float = 1e-8) -> Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    gram = centered @ centered.transpose(0, 1)
    return gram / gram.norm().clamp_min(epsilon)


def relational_loss(student: Tensor, teacher: Tensor) -> Tensor:
    losses = []
    for layer in range(student.shape[1]):
        student_gram = normalized_gram(student[:, layer])
        teacher_gram = normalized_gram(teacher[:, layer])
        losses.append((student_gram - teacher_gram).pow(2).sum())
    return torch.stack(losses).mean()


def pointwise_loss(student: Tensor, teacher: Tensor) -> Tensor:
    student_normalized = F.normalize(student, dim=-1)
    teacher_normalized = F.normalize(teacher, dim=-1)
    return (1.0 - (student_normalized * teacher_normalized).sum(dim=-1)).mean()


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    temperature: float,
    label_weight: float,
    logit_weight: float,
) -> Tensor:
    label_loss = F.cross_entropy(student_logits, labels)
    teacher_probabilities = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probabilities = F.log_softmax(
        student_logits / temperature, dim=-1
    )
    logit_loss = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * (temperature**2)
    return label_weight * label_loss + logit_weight * logit_loss


def train_student(
    model: nn.Module,
    dataset: CompositionDataset,
    *,
    condition: str,
    teacher_logits: Tensor,
    teacher_trajectory: Tensor,
    config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    loader = make_loader(
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
    peak_memory = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(int(config["epochs"])):
        for indices, tokens, labels, _ in loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_teacher_logits = teacher_logits[indices].to(
                device, non_blocking=True
            )
            batch_teacher_trajectory = teacher_trajectory[indices].to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)
            student_logits, student_trajectory = model(tokens)
            if condition == "labels":
                base_loss = F.cross_entropy(student_logits, labels)
            else:
                base_loss = distillation_loss(
                    student_logits,
                    batch_teacher_logits,
                    labels,
                    temperature=float(training_config["temperature"]),
                    label_weight=float(training_config["label_weight"]),
                    logit_weight=float(training_config["logit_weight"]),
                )

            auxiliary = torch.zeros((), device=device)
            if condition == "pointwise":
                auxiliary = pointwise_loss(
                    student_trajectory, batch_teacher_trajectory
                )
            elif condition in {"relational", "shuffled"}:
                target = batch_teacher_trajectory
                if condition == "shuffled":
                    target = target[torch.randperm(target.shape[0], device=device)]
                auxiliary = relational_loss(student_trajectory, target)

            loss = base_loss + float(training_config["auxiliary_weight"]) * auxiliary
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config["gradient_clip"])
            )
            optimizer.step()
            last_loss = float(loss.detach().item())
            last_base_loss = float(base_loss.detach().item())
            last_auxiliary_loss = float(auxiliary.detach().item())

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())
    return {
        "train_seconds": time.perf_counter() - start,
        "last_batch_loss": last_loss,
        "last_base_loss": last_base_loss,
        "last_auxiliary_loss": last_auxiliary_loss,
        "peak_gpu_memory_bytes": peak_memory,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["condition"])].append(run)

    summary = []
    for condition, condition_runs in grouped.items():
        row: dict[str, Any] = {"condition": condition, "runs": len(condition_runs)}
        for metric in (
            "id_accuracy",
            "ood_accuracy",
            "id_cross_entropy",
            "ood_cross_entropy",
            "train_seconds",
        ):
            values = np.asarray(
                [float(run[metric]) for run in condition_runs], dtype=np.float64
            )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return sorted(summary, key=lambda item: str(item["condition"]))


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
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
    data_config = config["data"]
    training_config = config["training"]
    all_depths = (
        list(data_config["teacher_depths"])
        + list(data_config["student_depths"])
        + list(data_config["id_depths"])
        + list(data_config["ood_depths"])
    )
    max_depth = max(all_depths)
    relation_table, relation_definitions = make_relation_table(
        int(data_config["num_entities"]),
        int(data_config["num_relations"]),
        int(config["data_seed"]),
    )
    teacher_train = CompositionDataset(
        count=int(data_config["teacher_train_examples"]),
        depths=list(data_config["teacher_depths"]),
        table=relation_table,
        seed=int(config["data_seed"]) + 1,
        max_depth=max_depth,
    )
    student_train = CompositionDataset(
        count=int(data_config["student_train_examples"]),
        depths=list(data_config["student_depths"]),
        table=relation_table,
        seed=int(config["data_seed"]) + 2,
        max_depth=max_depth,
        forbidden_pairs=list(data_config["heldout_pairs"]),
    )
    id_test = CompositionDataset(
        count=int(data_config["id_test_examples"]),
        depths=list(data_config["id_depths"]),
        table=relation_table,
        seed=int(config["data_seed"]) + 3,
        max_depth=max_depth,
        forbidden_pairs=list(data_config["heldout_pairs"]),
    )
    ood_test = CompositionDataset(
        count=int(data_config["ood_test_examples"]),
        depths=list(data_config["ood_depths"]),
        table=relation_table,
        seed=int(config["data_seed"]) + 4,
        max_depth=max_depth,
        required_pairs=list(data_config["heldout_pairs"]),
    )

    set_seed(int(config["teacher_seed"]))
    teacher = model_from_config(config["teacher"], teacher_train).to(device)
    teacher_parameters = count_parameters(teacher)
    teacher_training = train_teacher(
        teacher,
        teacher_train,
        config=config["teacher"],
        training_config=training_config,
        seed=int(config["teacher_seed"]),
        device=device,
    )
    teacher_id = evaluate(
        teacher,
        id_test,
        device=device,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
    )
    teacher_ood = evaluate(
        teacher,
        ood_test,
        device=device,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
    )

    student_layers = int(config["student"]["n_layers"])
    teacher_logits, teacher_trajectory, selected_layers = cache_teacher(
        teacher,
        student_train,
        student_layers=student_layers,
        device=device,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
    )

    runs: list[dict[str, Any]] = []
    student_parameters = 0
    for seed in config["student_seeds"]:
        for condition in config["conditions"]:
            set_seed(int(seed))
            student = model_from_config(config["student"], student_train).to(device)
            student_parameters = count_parameters(student)
            training_metrics = train_student(
                student,
                student_train,
                condition=str(condition),
                teacher_logits=teacher_logits,
                teacher_trajectory=teacher_trajectory,
                config=config["student"],
                training_config=training_config,
                seed=int(seed),
                device=device,
            )
            id_metrics = evaluate(
                student,
                id_test,
                device=device,
                batch_size=int(training_config["batch_size"]),
                num_workers=int(training_config["num_workers"]),
            )
            ood_metrics = evaluate(
                student,
                ood_test,
                device=device,
                batch_size=int(training_config["batch_size"]),
                num_workers=int(training_config["num_workers"]),
            )
            runs.append(
                {
                    "condition": str(condition),
                    "seed": int(seed),
                    "id_accuracy": id_metrics.accuracy,
                    "ood_accuracy": ood_metrics.accuracy,
                    "id_cross_entropy": id_metrics.cross_entropy,
                    "ood_cross_entropy": ood_metrics.cross_entropy,
                    "id_accuracy_by_depth": id_metrics.accuracy_by_depth,
                    "ood_accuracy_by_depth": ood_metrics.accuracy_by_depth,
                    **training_metrics,
                }
            )
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "seed": seed,
                        "id_accuracy": id_metrics.accuracy,
                        "ood_accuracy": ood_metrics.accuracy,
                        "train_seconds": training_metrics["train_seconds"],
                    }
                ),
                flush=True,
            )
            del student
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize_runs(runs)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": config["experiment_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pilot",
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain=v1"),
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
            "gpu_power_limit_w": nvidia_value("power.limit"),
            "gpu_power_draw_w": nvidia_value("power.draw"),
        },
        "relation_definitions": relation_definitions,
        "teacher": {
            "parameters": teacher_parameters,
            "selected_trajectory_layers_zero_based": selected_layers,
            "training": teacher_training,
            "id": asdict(teacher_id),
            "ood": asdict(teacher_ood),
        },
        "student_parameters": student_parameters,
        "runs": runs,
        "summary": summary,
        "limitations": [
            "Synthetic affine-composition task, not natural language.",
            "Pilot conditions do not include every E001 preregistered control.",
            "GPU power draw is unavailable under the current WDDM telemetry path.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(output_path.with_suffix(".csv"), summary)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
