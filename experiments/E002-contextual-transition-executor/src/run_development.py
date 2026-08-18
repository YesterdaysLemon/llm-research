from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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
P0_PATH = ROOT / "experiments" / "E001-trajectory-sufficiency" / "src" / "run_pilot.py"
P0_SPEC = importlib.util.spec_from_file_location("e002_p0", P0_PATH)
assert P0_SPEC is not None and P0_SPEC.loader is not None
p0 = importlib.util.module_from_spec(P0_SPEC)
sys.modules[P0_SPEC.name] = p0
P0_SPEC.loader.exec_module(p0)


class TransitionTableExecutor(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, pad_id: int) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.pad_id = pad_id
        self.transition_logits = nn.Parameter(
            torch.zeros(num_relations, num_entities, num_entities)
        )

    @property
    def active_parameters_per_step(self) -> int:
        return self.num_entities * self.num_entities

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        start = tokens[:, 1]
        state = F.one_hot(start, num_classes=self.num_entities).to(torch.float32)
        transition_probabilities = self.transition_logits.softmax(dim=-1)
        trajectory: list[Tensor] = []
        for position in range(2, tokens.shape[1]):
            relation = tokens[:, position] - self.num_entities
            valid = relation.ge(0) & relation.lt(self.num_relations)
            selected = transition_probabilities[relation.clamp(0, self.num_relations - 1)]
            updated = torch.bmm(state.unsqueeze(1), selected).squeeze(1)
            state = torch.where(valid.unsqueeze(1), updated, state)
            trajectory.append(state)
        logits = state.clamp_min(1e-12).log()
        return logits, torch.stack(trajectory, dim=1)


class GRUExecutor(nn.Module):
    def __init__(
        self, num_entities: int, num_relations: int, pad_id: int, d_model: int
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.pad_id = pad_id
        self.entity_embedding = nn.Embedding(num_entities, d_model)
        self.relation_embedding = nn.Embedding(num_relations, d_model)
        self.cell = nn.GRUCell(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_entities)

    @property
    def active_parameters_per_step(self) -> int:
        return sum(parameter.numel() for parameter in self.cell.parameters())

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        state = self.entity_embedding(tokens[:, 1])
        trajectory: list[Tensor] = []
        for position in range(2, tokens.shape[1]):
            relation = tokens[:, position] - self.num_entities
            valid = relation.ge(0) & relation.lt(self.num_relations)
            relation_features = self.relation_embedding(
                relation.clamp(0, self.num_relations - 1)
            )
            updated = self.cell(relation_features, state)
            state = torch.where(valid.unsqueeze(1), updated, state)
            trajectory.append(state)
        return self.classifier(state), torch.stack(trajectory, dim=1)


def build_model(spec: dict[str, Any], dataset: Any) -> nn.Module:
    model_type = str(spec["type"])
    if model_type == "transition_table":
        return TransitionTableExecutor(
            dataset.num_entities, dataset.num_relations, dataset.pad_id
        )
    if model_type == "gru":
        return GRUExecutor(
            dataset.num_entities,
            dataset.num_relations,
            dataset.pad_id,
            int(spec["d_model"]),
        )
    if model_type == "transformer":
        return p0.model_from_config(spec, dataset)
    raise ValueError(f"unknown model type: {model_type}")


def active_parameters(model: nn.Module) -> int:
    value = getattr(model, "active_parameters_per_step", None)
    if value is None:
        return p0.count_parameters(model)
    return int(value)


def train(
    model: nn.Module,
    dataset: Any,
    *,
    spec: dict[str, Any],
    training: dict[str, Any],
    seed: int,
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    steps = 0
    last_loss = float("nan")
    for _ in range(int(spec["epochs"])):
        for _, tokens, labels, _ in loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tokens)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            steps += 1
            last_loss = float(loss.detach().item())
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
        "peak_gpu_memory_bytes": peak_memory,
    }


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["model_id"])].append(run)
    output: list[dict[str, Any]] = []
    for model_id, rows in grouped.items():
        summary: dict[str, Any] = {
            "model_id": model_id,
            "model_type": rows[0]["model_type"],
            "stored_parameters": rows[0]["stored_parameters"],
            "active_parameters_per_step": rows[0]["active_parameters_per_step"],
            "runs": len(rows),
        }
        for metric in ("id_accuracy", "evaluation_accuracy", "train_seconds"):
            values = np.asarray([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        output.append(summary)
    return sorted(output, key=lambda item: str(item["model_id"]))


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

    if config["task_family"] != "affine":
        raise ValueError("development runner currently supports affine tasks")
    data = config["data"]
    table, definitions = p0.make_relation_table(
        int(data["num_entities"]),
        int(data["num_relations"]),
        int(config["data_seed"]),
    )
    max_depth = max(
        list(data["train_depths"])
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
        seed=int(config["data_seed"]) + 3,
        max_depth=max_depth,
        forbidden_pairs=list(data["evaluation_forbidden_pairs"]),
        required_pairs=list(data["evaluation_pairs"]),
    )

    runs: list[dict[str, Any]] = []
    for seed in config["student_seeds"]:
        for spec in config["models"]:
            p0.set_seed(int(seed))
            model = build_model(spec, train_data).to(device)
            stored = p0.count_parameters(model)
            active = active_parameters(model)
            training_metrics = train(
                model,
                train_data,
                spec=spec,
                training=config["training"],
                seed=int(seed),
                device=device,
            )
            id_metrics = p0.evaluate(
                model,
                id_data,
                device=device,
                batch_size=int(config["training"]["batch_size"]),
                num_workers=int(config["training"]["num_workers"]),
            )
            evaluation_metrics = p0.evaluate(
                model,
                evaluation_data,
                device=device,
                batch_size=int(config["training"]["batch_size"]),
                num_workers=int(config["training"]["num_workers"]),
            )
            row = {
                "model_id": spec["id"],
                "model_type": spec["type"],
                "seed": int(seed),
                "stored_parameters": stored,
                "active_parameters_per_step": active,
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
                        "model_id": spec["id"],
                        "seed": seed,
                        "stored_parameters": stored,
                        "id_accuracy": id_metrics.accuracy,
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
        "runs": runs,
        "summary": summary,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), summary)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
