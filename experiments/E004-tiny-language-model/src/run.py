from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data import (
    PairedBatchStream,
    Vocabulary,
    build_vocabulary,
    encode_controlled_records,
    generate_controlled_records,
    generate_eval_records,
    load_story_split,
    pack_stories,
    sha256_file,
)
from model import count_parameters, learning_rate_multiplier, model_from_config


ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def autocast_context(device: torch.device, enabled: bool) -> Any:
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    source = ROOT / dataset["path"]
    if not source.exists():
        raise FileNotFoundError(f"missing dataset; run fetch_data.py: {source}")
    if source.stat().st_size != int(dataset["bytes"]):
        raise RuntimeError("dataset byte count does not match frozen config")
    source_hash = sha256_file(source)
    if source_hash.lower() != str(dataset["sha256"]).lower():
        raise RuntimeError("dataset SHA-256 does not match frozen config")
    train_stories, validation_stories = load_story_split(
        source,
        train_stories=int(dataset["train_stories"]),
        validation_stories=int(dataset["validation_stories"]),
    )
    vocabulary = build_vocabulary(train_stories, int(dataset["vocab_size"]))
    controlled_records = generate_controlled_records(
        count=int(dataset["controlled_train_records"]),
        depths=dataset["controlled_train_depths"],
        seed=int(dataset["controlled_seed"]),
        forbidden_pairs=dataset["heldout_pairs"],
    )
    return {
        "source_hash": source_hash,
        "vocabulary": vocabulary,
        "natural_train": pack_stories(train_stories, vocabulary),
        "natural_validation": pack_stories(validation_stories, vocabulary),
        "controlled_train": encode_controlled_records(
            controlled_records, vocabulary
        ),
    }


def masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    answer_mask: Tensor,
    *,
    answer_weight: float,
) -> Tensor:
    losses = F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(targets)
    weights = mask.to(losses.dtype)
    weights = weights * torch.where(
        answer_mask,
        torch.as_tensor(answer_weight, dtype=losses.dtype, device=losses.device),
        torch.ones((), dtype=losses.dtype, device=losses.device),
    )
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def accumulate_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def train_capability_ladder(
    model: nn.Module,
    *,
    checkpoint_steps: list[int],
    seed: int,
    data: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    evaluate: Callable[[nn.Module], dict[str, Any]],
    gate: Callable[[dict[str, Any]], dict[str, bool]],
) -> dict[str, Any]:
    set_seed(seed)
    dataset = config["dataset"]
    training = config["training"]
    stream = PairedBatchStream(
        data["natural_train"],
        data["controlled_train"],
        sequence_length=int(dataset["sequence_length"]),
        controlled_fraction=float(dataset["controlled_sequence_fraction"]),
        pad_id=int(data["vocabulary"].stoi["<pad>"]),
        question_id=int(data["vocabulary"].stoi["?"]),
        seed=seed + 100_000,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    maximum_steps = max(checkpoint_steps)
    if checkpoint_steps != sorted(set(checkpoint_steps)):
        raise ValueError("checkpoint steps must be sorted and unique")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    cumulative_loss = 0.0
    cumulative_training_seconds = 0.0
    source_totals: dict[str, int] = {}
    rungs: list[dict[str, Any]] = []
    selected_step: int | None = None
    segment_started = time.perf_counter()
    for step_index in range(maximum_steps):
        step = step_index + 1
        multiplier = learning_rate_multiplier(
            step_index,
            total_steps=maximum_steps,
            warmup_steps=int(training["warmup_steps"]),
            minimum_ratio=float(training["min_learning_rate_ratio"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = float(training["learning_rate"]) * multiplier
        inputs, targets, mask, answer_mask, metadata = stream.batch(
            int(training["batch_size"])
        )
        accumulate_counts(source_totals, metadata)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        answer_mask = answer_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bool(training["use_bfloat16"])):
            logits, _ = model(inputs)
            loss = masked_cross_entropy(
                logits,
                targets,
                mask,
                answer_mask,
                answer_weight=float(training.get("answer_loss_weight", 1.0)),
            )
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip"])
        )
        optimizer.step()
        cumulative_loss += float(loss.detach().item())
        if step not in checkpoint_steps:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        cumulative_training_seconds += time.perf_counter() - segment_started
        evaluation_started = time.perf_counter()
        evaluation = evaluate(model)
        evaluation_seconds = time.perf_counter() - evaluation_started
        checks = gate(evaluation)
        rung = {
            "step": step,
            "mean_training_loss": cumulative_loss / step,
            "training_seconds_cumulative": cumulative_training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "source_exposures": dict(source_totals),
            "evaluation": evaluation,
            "gates": checks,
        }
        rungs.append(rung)
        print(
            json.dumps(
                {
                    "event": "capability_rung",
                    "step": step,
                    "mean_training_loss": cumulative_loss / step,
                    "source_exposures": source_totals,
                    "gates": checks,
                }
            ),
            flush=True,
        )
        if checks["all"]:
            selected_step = step
            break
        model.train()
        segment_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())
    else:
        peak_memory = 0
    return {
        "checkpoint_steps": checkpoint_steps,
        "selected_step": selected_step,
        "peak_gpu_memory_bytes": peak_memory,
        "rungs": rungs,
    }


@torch.no_grad()
def evaluate_natural(
    model: nn.Module,
    stream: Tensor,
    *,
    sequence_length: int,
    batches: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for index in range(batches):
        start = index * sequence_length
        stop = start + sequence_length + 1
        if stop > stream.numel():
            raise ValueError("natural validation stream is too short")
        values = stream[start:stop].unsqueeze(0).to(device)
        logits, _ = model(values[:, :-1])
        loss = F.cross_entropy(
            logits.flatten(0, 1), values[:, 1:].flatten(), reduction="sum"
        )
        total_loss += float(loss.item())
        total_tokens += sequence_length
    nll = total_loss / total_tokens
    return {
        "nll": nll,
        "perplexity": math.exp(min(20.0, nll)),
        "tokens": total_tokens,
        "precision": "float32",
    }


@torch.no_grad()
def evaluate_controlled(
    model: nn.Module,
    vocabulary: Vocabulary,
    rows: Iterable[tuple[int, str, str, int]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    lookup = vocabulary.stoi
    correct_by_depth: defaultdict[int, int] = defaultdict(int)
    counts_by_depth: defaultdict[int, int] = defaultdict(int)
    correct_by_pair: defaultdict[str, int] = defaultdict(int)
    counts_by_pair: defaultdict[str, int] = defaultdict(int)
    total_nll = 0.0
    for depth, pair_label, prompt, answer in rows:
        tokens = vocabulary.encode(prompt)
        if len(tokens) > model.sequence_length:
            raise ValueError("controlled prompt exceeds model context")
        inputs = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(inputs)
        final_logits = logits[0, -1].float()
        answer_id = lookup[str(answer)]
        predicted = int(final_logits.argmax().item())
        correct_by_depth[int(depth)] += int(predicted == answer_id)
        counts_by_depth[int(depth)] += 1
        correct_by_pair[pair_label] += int(predicted == answer_id)
        counts_by_pair[pair_label] += 1
        total_nll += float(
            F.cross_entropy(
                final_logits.unsqueeze(0),
                torch.tensor([answer_id], device=device),
            ).item()
        )
    total = sum(counts_by_depth.values())
    return {
        "accuracy": sum(correct_by_depth.values()) / total,
        "answer_nll": total_nll / total,
        "examples": total,
        "accuracy_by_depth": {
            str(depth): correct_by_depth[depth] / counts_by_depth[depth]
            for depth in sorted(counts_by_depth)
        },
        "examples_by_depth": {
            str(depth): counts_by_depth[depth] for depth in sorted(counts_by_depth)
        },
        "accuracy_by_pair": {
            pair: correct_by_pair[pair] / counts_by_pair[pair]
            for pair in sorted(counts_by_pair)
        },
        "examples_by_pair": {
            pair: counts_by_pair[pair] for pair in sorted(counts_by_pair)
        },
        "precision": "float32",
    }


def unigram_validation_nll(
    train_stream: Tensor,
    validation_stream: Tensor,
    *,
    vocab_size: int,
    sequence_length: int,
    batches: int,
) -> float:
    counts = torch.bincount(train_stream, minlength=vocab_size).double() + 1.0
    log_probabilities = (counts / counts.sum()).log()
    targets: list[Tensor] = []
    for index in range(batches):
        start = index * sequence_length
        targets.append(validation_stream[start + 1 : start + sequence_length + 1])
    values = torch.cat(targets)
    return float((-log_probabilities[values]).mean().item())


def evaluate_model(
    model: nn.Module,
    data: dict[str, Any],
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    dataset = config["dataset"]
    count = int(evaluation["controlled_examples"])
    id_rows = generate_eval_records(
        count=count,
        depths=evaluation["id_depths"],
        seed=int(evaluation["seed"]),
        forbidden_pairs=dataset["heldout_pairs"],
        require_heldout=False,
    )
    heldout_rows = generate_eval_records(
        count=count,
        depths=evaluation["heldout_depths"],
        seed=int(evaluation["seed"]) + 1,
        forbidden_pairs=dataset["heldout_pairs"],
        require_heldout=True,
    )
    return {
        "natural": evaluate_natural(
            model,
            data["natural_validation"],
            sequence_length=int(dataset["sequence_length"]),
            batches=int(evaluation["natural_batches"]),
            device=device,
        ),
        "controlled_id": evaluate_controlled(
            model, data["vocabulary"], id_rows, device=device
        ),
        "controlled_heldout": evaluate_controlled(
            model, data["vocabulary"], heldout_rows, device=device
        ),
    }


def mean_depths(metrics: dict[str, Any], depths: Iterable[int]) -> float:
    values = metrics["accuracy_by_depth"]
    selected = [float(values[str(depth)]) for depth in depths]
    return sum(selected) / len(selected)


def teacher_gate(
    evaluation: dict[str, Any], *, baseline_nll: float, gates: dict[str, Any]
) -> dict[str, bool]:
    teacher_id = evaluation["controlled_id"]
    heldout = evaluation["controlled_heldout"]
    checks = {
        "natural_beats_unigram": baseline_nll - evaluation["natural"]["nll"]
        >= float(gates["natural_nll_improvement_min"]),
        "id_overall": teacher_id["accuracy"]
        >= float(gates["teacher_id_accuracy_min"]),
        "id_each_depth": min(teacher_id["accuracy_by_depth"].values())
        >= float(gates["teacher_id_accuracy_each_depth_min"]),
        "heldout_depth_2_to_4": mean_depths(heldout, (2, 3, 4))
        >= float(gates["teacher_heldout_depth_2_to_4_accuracy_min"]),
        "heldout_each_pair": min(heldout["accuracy_by_pair"].values())
        >= float(gates["teacher_heldout_each_pair_accuracy_min"]),
    }
    return {**checks, "all": all(checks.values())}


def student_gate(
    evaluation: dict[str, Any], *, baseline_nll: float, gates: dict[str, Any]
) -> dict[str, bool]:
    student_id = evaluation["controlled_id"]
    heldout = evaluation["controlled_heldout"]
    heldout_primary = mean_depths(heldout, (2, 3, 4))
    checks = {
        "natural_beats_unigram": baseline_nll - evaluation["natural"]["nll"]
        >= float(gates["natural_nll_improvement_min"]),
        "id_overall": student_id["accuracy"]
        >= float(gates["student_id_accuracy_min"]),
        "id_depth_4": student_id["accuracy_by_depth"]["4"]
        >= float(gates["student_id_depth_4_accuracy_min"]),
        "heldout_above_floor": heldout_primary
        >= float(gates["student_heldout_depth_2_to_4_accuracy_min"]),
        "heldout_below_ceiling": heldout_primary
        <= float(gates["student_heldout_depth_2_to_4_accuracy_max"]),
    }
    return {**checks, "all": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    configure_determinism()
    torch.set_float32_matmul_precision("high")
    data = prepare_data(config)
    vocabulary = data["vocabulary"]
    dataset = config["dataset"]
    evaluation = config["evaluation"]
    baseline_nll = unigram_validation_nll(
        data["natural_train"],
        data["natural_validation"],
        vocab_size=len(vocabulary.tokens),
        sequence_length=int(dataset["sequence_length"]),
        batches=int(evaluation["natural_batches"]),
    )

    set_seed(int(config["teacher"]["seed"]))
    teacher = model_from_config(
        config["teacher"],
        vocab_size=len(vocabulary.tokens),
        sequence_length=int(dataset["sequence_length"]),
    ).to(device)
    teacher_training = train_capability_ladder(
        teacher,
        checkpoint_steps=[int(value) for value in config["smoke"]["checkpoint_steps"]],
        seed=int(config["teacher"]["seed"]),
        data=data,
        config=config,
        device=device,
        evaluate=lambda model: evaluate_model(model, data, config, device=device),
        gate=lambda metrics: teacher_gate(
            metrics, baseline_nll=baseline_nll, gates=config["gates"]
        ),
    )

    set_seed(int(config["student"]["seed"]))
    student = model_from_config(
        config["student"],
        vocab_size=len(vocabulary.tokens),
        sequence_length=int(dataset["sequence_length"]),
    ).to(device)
    student_training = train_capability_ladder(
        student,
        checkpoint_steps=[int(value) for value in config["smoke"]["checkpoint_steps"]],
        seed=int(config["student"]["seed"]),
        data=data,
        config=config,
        device=device,
        evaluate=lambda model: evaluate_model(model, data, config, device=device),
        gate=lambda metrics: student_gate(
            metrics, baseline_nll=baseline_nll, gates=config["gates"]
        ),
    )

    result: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "phase": config["phase"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain=v1"),
        "dataset": {
            "path": dataset["path"],
            "revision": dataset["revision"],
            "license": dataset["license"],
            "bytes": int(dataset["bytes"]),
            "sha256": data["source_hash"],
            "vocabulary_size": len(vocabulary.tokens),
            "vocabulary_sha256": hashlib.sha256(
                "\n".join(vocabulary.tokens).encode("utf-8")
            ).hexdigest(),
            "natural_train_tokens": int(data["natural_train"].numel()),
            "natural_validation_tokens": int(data["natural_validation"].numel()),
            "controlled_train_records": len(data["controlled_train"]),
            "controlled_train_tokens": sum(
                int(record.numel()) for record in data["controlled_train"]
            ),
            "unigram_validation_nll": baseline_nll,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "training_autocast": (
                "bfloat16"
                if device.type == "cuda" and config["training"]["use_bfloat16"]
                else "float32"
            ),
            "evaluation_precision": "float32",
        },
        "teacher": {
            "parameters": count_parameters(teacher),
            "training": teacher_training,
        },
        "student": {
            "parameters": count_parameters(student),
            "training": student_training,
        },
        "gates": {
            "teacher_selected_step": teacher_training["selected_step"],
            "student_selected_step": student_training["selected_step"],
            "all": teacher_training["selected_step"] is not None
            and student_training["selected_step"] is not None,
        },
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "gates": result["gates"]}), flush=True)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
