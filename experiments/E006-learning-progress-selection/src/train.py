"""Run the E006 selection arms and write one raw result file per run."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates import (  # noqa: E402
    CandidatePool,
    prepare_natural_data,
    sha256_file,
    unigram_nll,
    validation_windows,
)
from model import count_parameters, learning_rate_multiplier, model_from_config  # noqa: E402  (E004)
from selection import (  # noqa: E402
    ARM_REQUIREMENT,
    ARMS,
    SnapshotBank,
    adam_step_operator,
    arm_scores,
    jvp_scores,
    learning_direction,
    parameter_dict,
    per_sample_statistics,
    sequence_losses,
    spearman,
    top_indices,
)

ROOT = Path(__file__).resolve().parents[3]

# Analytic cost convention, in multiples of one forward pass over the same tokens.
REQUIREMENT_COST = {"none": 0, "forward": 1, "jvp": 2, "per_sample": 3}
TRAIN_COST = 3


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
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def autocast_context(device: torch.device, enabled: bool) -> Any:
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def forward_flops_per_token(spec: dict[str, Any], *, vocab_size: int, sequence_length: int) -> float:
    """Kaplan-style forward FLOPs per token: weight matmuls, tied LM head, attention context."""
    width = int(spec["d_model"])
    layers = int(spec["n_layers"])
    feed_forward = int(spec["d_ff"])
    block_weights = 4 * width * width + 2 * width * feed_forward
    return float(
        2 * layers * block_weights
        + 2 * width * vocab_size
        + 2 * layers * sequence_length * width
    )


def step_flops(arm: str, *, forward_per_token: float, batch: int, pool: int, length: int, scored: bool) -> float:
    train = TRAIN_COST * forward_per_token * batch * length
    if not scored:
        return train
    return train + REQUIREMENT_COST[ARM_REQUIREMENT[arm]] * forward_per_token * pool * length


def flops_matched_steps(selection_steps: int, *, batch: int, pool: int) -> int:
    """Uniform steps whose nominal FLOPs equal `selection_steps` of the JVP arms."""
    ratio = (TRAIN_COST * batch + REQUIREMENT_COST["jvp"] * pool) / (TRAIN_COST * batch)
    return int(math.ceil(selection_steps * ratio))


@torch.no_grad()
def validation_nll(model: nn.Module, windows: Tensor, *, device: torch.device, batch: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, windows.shape[0], batch):
        values = windows[start : start + batch].to(device)
        logits, _ = model(values[:, :-1])
        losses = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), values[:, 1:].flatten(), reduction="sum"
        )
        total += float(losses.item())
        count += values[:, 1:].numel()
    model.train()
    return total / count


def pool_statistics(
    requirement: str,
    *,
    model: nn.Module,
    params: dict[str, Tensor],
    direction: Any,
    inputs: Tensor,
    targets: Tensor,
    chunk: int,
) -> dict[str, Tensor]:
    if requirement == "forward":
        with torch.no_grad():
            return {"loss": sequence_losses(model, params, inputs, targets).detach()}
    if requirement == "jvp":
        losses, derivative = jvp_scores(model, params, direction.tangent, inputs, targets)
        return {"loss": losses, "jvp": derivative}
    if requirement == "per_sample":
        return per_sample_statistics(model, params, direction, inputs, targets, chunk=chunk)
    raise ValueError(f"unknown requirement {requirement!r}")


def diagnostics(
    *,
    step: int,
    model: nn.Module,
    params: dict[str, Tensor],
    direction: Any,
    inputs: Tensor,
    targets: Tensor,
    noise_mask: Tensor,
    chunk: int,
) -> dict[str, Any]:
    """All five scores on the current pool; never used for selection."""
    losses, derivative = jvp_scores(model, params, direction.tangent, inputs, targets)
    sample = per_sample_statistics(model, params, direction, inputs, targets, chunk=chunk)
    statistics = {**sample, "jvp": derivative}
    scores = {
        arm: arm_scores(arm, statistics, direction.preconditioned_norm).cpu()
        for arm in ARMS
        if arm != "uniform"
    }
    names = list(scores)
    correlations = {
        f"{left}|{right}": spearman(scores[left], scores[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    scale = float(derivative.abs().max().item())
    error = float((derivative - sample["dot"]).abs().max().item())
    noise = noise_mask.cpu()
    by_source: dict[str, Any] = {}
    for label, mask in (("clean", ~noise), ("noise", noise)):
        if bool(mask.any()):
            by_source[label] = {
                arm: float(values[mask].mean().item()) for arm, values in scores.items()
            }
    return {
        "step": int(step),
        "anchor_step": direction.anchor_step,
        "direction_norm": direction.preconditioned_norm,
        "spearman": correlations,
        "jvp_vs_per_sample_max_abs_error": error,
        "jvp_max_abs": scale,
        "jvp_primal_loss_max_abs_error": float((losses - sample["loss"]).abs().max().item()),
        "noise_windows_in_pool": int(noise.sum().item()),
        "mean_score_by_source": by_source,
    }


def run_one(
    *,
    condition: dict[str, Any],
    arm: str,
    seed: int,
    total_steps: int,
    evaluation_steps: list[int],
    data: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    dataset, training, selection = config["dataset"], config["training"], config["selection"]
    length = int(dataset["sequence_length"])
    batch = int(training["batch_size"])
    pool_size = int(selection["pool_multiple"]) * batch
    chunk = int(selection["per_sample_chunk"])
    diagnostic_interval = int(selection["diagnostic_interval"])
    vocab_size = len(data["vocabulary"].tokens)
    forward_per_token = forward_flops_per_token(
        config["student"], vocab_size=vocab_size, sequence_length=length
    )

    set_seed(seed)
    model = model_from_config(
        config["student"], vocab_size=vocab_size, sequence_length=length
    ).to(device)
    params = parameter_dict(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    pool = CandidatePool(
        data["train"],
        sequence_length=length,
        vocab_size=vocab_size,
        seed=seed + int(selection["pool_seed_offset"]),
        noise_fraction=float(condition["noise_fraction"]),
        noise_seed=seed + int(selection["noise_seed_offset"]),
    )
    bank = SnapshotBank(int(selection["snapshot_interval"]))
    bank.store(0, params)
    windows = data["validation_windows"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    evaluations: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    seconds = {"scoring": 0.0, "training": 0.0, "diagnostics": 0.0, "evaluation": 0.0}
    cumulative_flops = 0.0
    selected_noise = 0
    selected_windows = 0
    nonfinite = False
    checkpoints = set(evaluation_steps)
    model.train()
    for step in range(total_steps):
        multiplier = learning_rate_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=int(training["warmup_steps"]),
            minimum_ratio=float(training["min_learning_rate_ratio"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = float(training["learning_rate"]) * multiplier
        inputs, targets, noise_mask = pool.draw(pool_size)
        inputs, targets = inputs.to(device), targets.to(device)

        # Every non-uniform arm selects from the second step on; the first step
        # has no optimizer state and trains on the first `batch` pool windows.
        scored = arm != "uniform" and step > 0
        method_needs_direction = scored and ARM_REQUIREMENT[arm] in ("jvp", "per_sample")
        diagnostic_due = step > 0 and step % diagnostic_interval == 0

        def current_direction() -> Any:
            anchor_step, anchor = bank.anchor(step, device)
            return learning_direction(
                params, anchor, adam_step_operator(optimizer, params), anchor_step
            )

        direction = None
        if diagnostic_due:
            synchronize(device)
            started = time.perf_counter()
            direction = current_direction()
            diagnostic_rows.append(
                diagnostics(
                    step=step,
                    model=model,
                    params=params,
                    direction=direction,
                    inputs=inputs,
                    targets=targets,
                    noise_mask=noise_mask,
                    chunk=chunk,
                )
            )
            synchronize(device)
            seconds["diagnostics"] += time.perf_counter() - started

        synchronize(device)
        started = time.perf_counter()
        if method_needs_direction and direction is None:
            direction = current_direction()
        if scored:
            direction_norm = direction.preconditioned_norm if direction is not None else 0.0
            statistics = pool_statistics(
                ARM_REQUIREMENT[arm],
                model=model,
                params=params,
                direction=direction,
                inputs=inputs,
                targets=targets,
                chunk=chunk,
            )
            chosen = top_indices(arm_scores(arm, statistics, direction_norm), batch)
        else:
            chosen = torch.arange(batch)
        synchronize(device)
        seconds["scoring"] += time.perf_counter() - started

        started = time.perf_counter()
        chosen_device = chosen.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bool(training["use_bfloat16"])):
            logits, _ = model(inputs.index_select(0, chosen_device))
            loss = torch.nn.functional.cross_entropy(
                logits.float().flatten(0, 1),
                targets.index_select(0, chosen_device).flatten(),
            )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        synchronize(device)
        seconds["training"] += time.perf_counter() - started
        if not math.isfinite(float(loss.detach().item())):
            nonfinite = True

        cumulative_flops += step_flops(
            arm,
            forward_per_token=forward_per_token,
            batch=batch,
            pool=pool_size,
            length=length,
            scored=scored,
        )
        selected_noise += int(noise_mask[chosen].sum().item())
        selected_windows += batch
        bank.store(step + 1, params)
        bank.prune(step + 1)

        if step + 1 in checkpoints:
            started = time.perf_counter()
            nll = validation_nll(
                model, windows, device=device, batch=int(config["evaluation"]["batch_size"])
            )
            seconds["evaluation"] += time.perf_counter() - started
            nonfinite = nonfinite or not math.isfinite(nll)
            evaluations.append(
                {
                    "step": step + 1,
                    "validation_nll": nll,
                    "method_flops": cumulative_flops,
                    "tokens_trained": (step + 1) * batch * length,
                    "noise_windows_selected": selected_noise,
                    "windows_selected": selected_windows,
                    "method_seconds": seconds["scoring"] + seconds["training"],
                }
            )
            print(
                json.dumps(
                    {
                        "event": "evaluation",
                        "condition": condition["id"],
                        "arm": arm,
                        "seed": seed,
                        "step": step + 1,
                        "validation_nll": round(nll, 5),
                    }
                ),
                flush=True,
            )

    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    return {
        "condition": condition["id"],
        "noise_fraction": float(condition["noise_fraction"]),
        "arm": arm,
        "seed": seed,
        "total_steps": total_steps,
        "parameters": count_parameters(model),
        "pool_size": pool_size,
        "batch_size": batch,
        "forward_flops_per_token": forward_per_token,
        "evaluations": evaluations,
        "diagnostics": diagnostic_rows,
        "seconds": seconds,
        "peak_gpu_memory_bytes": peak,
        "nonfinite": nonfinite,
    }


def load_data(config: dict[str, Any]) -> dict[str, Any]:
    data = prepare_natural_data(config["dataset"])
    length = int(config["dataset"]["sequence_length"])
    data["validation_windows"] = validation_windows(data["validation"], length)
    data["unigram_validation_nll"] = unigram_nll(
        data["train"], data["validation_windows"], len(data["vocabulary"].tokens)
    )
    return data


def schedule(config: dict[str, Any]) -> list[tuple[dict[str, Any], str, int, int, list[int]]]:
    """Every (condition, arm, seed, total steps, evaluation steps) in run order."""
    selection_steps = int(config["selection_steps"])
    uniform_steps = int(config["uniform_steps"])
    expected = flops_matched_steps(
        selection_steps,
        batch=int(config["training"]["batch_size"]),
        pool=int(config["selection"]["pool_multiple"]) * int(config["training"]["batch_size"]),
    )
    if uniform_steps != expected:
        raise ValueError(f"uniform_steps {uniform_steps} != FLOP-matched {expected}")
    if float(config["training"]["min_learning_rate_ratio"]) != 1.0:
        # A constant post-warmup rate makes the longer uniform run's first
        # `selection_steps` identical to a run stopped at that budget.
        raise ValueError("E006 requires a constant post-warmup learning rate")
    if list(config["arms"]) != list(ARMS):
        raise ValueError("config arms must be exactly the registered arms, in order")
    shared = sorted(int(step) for step in config["evaluation"]["steps"])
    extended = sorted(set(shared) | {int(step) for step in config["evaluation"]["uniform_extra_steps"]})
    if max(shared) != selection_steps or max(extended) != uniform_steps:
        raise ValueError("evaluation steps must end at the step budgets")
    runs = []
    for seed in config["seeds"]:
        for condition in config["conditions"]:
            for arm in config["arms"]:
                if arm == "uniform":
                    runs.append((condition, arm, int(seed), uniform_steps, extended))
                else:
                    runs.append((condition, arm, int(seed), selection_steps, shared))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
    # Full float32 matmuls (no TF32) so scores and the JVP consistency gate are exact
    # to float32 precision; bfloat16 autocast still applies to the training step.
    torch.set_float32_matmul_precision("highest")
    runs = schedule(config)
    data = load_data(config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        recorded_config_path = str(config_path.relative_to(ROOT))
    except ValueError:
        recorded_config_path = str(config_path)
    header = {
        "experiment_id": config["experiment_id"],
        "phase": config["phase"],
        "config_path": recorded_config_path,
        "config_sha256": sha256_file(config_path),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain=v1"),
        "dataset": {
            "path": config["dataset"]["path"],
            "revision": config["dataset"]["revision"],
            "license": config["dataset"]["license"],
            "sha256": data["source_hash"],
            "vocabulary_size": len(data["vocabulary"].tokens),
            "vocabulary_sha256": data["vocabulary_sha256"],
            "train_tokens": int(data["train"].numel()),
            "validation_tokens": int(data["validation"].numel()),
            "validation_windows": int(data["validation_windows"].shape[0]),
            "unigram_validation_nll": data["unigram_validation_nll"],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "attention_backend": "math",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "training_autocast": (
                "bfloat16"
                if device.type == "cuda" and config["training"]["use_bfloat16"]
                else "float32"
            ),
            "scoring_and_evaluation_precision": "float32",
        },
    }
    for condition, arm, seed, total_steps, evaluation_steps in runs:
        target = output_dir / f"{condition['id']}__{arm}__seed{seed}.json"
        if target.exists():
            print(json.dumps({"event": "skip_existing", "file": target.name}), flush=True)
            continue
        with sdpa_kernel(SDPBackend.MATH):
            result = run_one(
                condition=condition,
                arm=arm,
                seed=seed,
                total_steps=total_steps,
                evaluation_steps=evaluation_steps,
                data=data,
                config=config,
                device=device,
            )
        payload = {
            **header,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run": result,
        }
        partial = target.with_suffix(".json.partial")
        partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        partial.replace(target)
        print(json.dumps({"event": "run_complete", "file": target.name}), flush=True)
    print(json.dumps({"event": "complete", "runs": len(runs)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
