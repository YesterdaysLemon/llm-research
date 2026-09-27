"""Run E007 arms and write one raw result file per (arm, seed)."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "E004-tiny-language-model" / "src"))

from model import count_parameters, learning_rate_multiplier, model_from_config  # noqa: E402  (E004)
from natural import (  # noqa: E402
    NaturalStream,
    content_range,
    prepare_natural_data,
    sha256_file,
    unigram_nll,
    validation_windows,
)
from probes import evaluate_probes, probe_prompts  # noqa: E402
from synthetic import SyntheticStream, _bank_chunk, build_program_bank  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
ARMS = ("scratch", "pcfg", "pcfg_shuffled", "programs", "programs_shuffled")
SOURCES = ("pcfg", "programs")


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


def new_model(config: dict[str, Any], seed: int, vocab_size: int, device: torch.device) -> nn.Module:
    set_seed(seed)
    return model_from_config(
        config["student"],
        vocab_size=vocab_size,
        sequence_length=int(config["dataset"]["sequence_length"]),
    ).to(device)


def new_optimizer(model: nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor | None,
    *,
    step: int,
    total_steps: int,
    training: dict[str, Any],
    device: torch.device,
) -> float:
    multiplier = learning_rate_multiplier(
        step,
        total_steps=total_steps,
        warmup_steps=int(training["warmup_steps"]),
        minimum_ratio=float(training["min_learning_rate_ratio"]),
    )
    for group in optimizer.param_groups:
        group["lr"] = float(training["learning_rate"]) * multiplier
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, bool(training["use_bfloat16"])):
        logits, _ = model(inputs.to(device))
        losses = F.cross_entropy(
            logits.float().flatten(0, 1), targets.to(device).flatten(), reduction="none"
        )
        if mask is None:
            loss = losses.mean()
        else:
            weights = mask.to(device).flatten().float()
            loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def masked_nll(model: nn.Module, inputs: Tensor, targets: Tensor, mask: Tensor, *, device: torch.device, batch: int) -> float:
    model.eval()
    total = 0.0
    count = 0.0
    for start in range(0, inputs.shape[0], batch):
        logits, _ = model(inputs[start : start + batch].to(device))
        losses = F.cross_entropy(
            logits.float().flatten(0, 1),
            targets[start : start + batch].to(device).flatten(),
            reduction="none",
        )
        weights = mask[start : start + batch].to(device).flatten().float()
        total += float((losses * weights).sum().item())
        count += float(weights.sum().item())
    model.train()
    return total / count


def validation_nll(model: nn.Module, windows: Tensor, *, device: torch.device, batch: int) -> float:
    inputs, targets = windows[:, :-1], windows[:, 1:]
    return masked_nll(model, inputs, targets, torch.ones_like(targets, dtype=torch.bool), device=device, batch=batch)


def synthetic_stream(
    source: str,
    config: dict[str, Any],
    seed: int,
    vocab_size: int,
    bank: list[list[int]] | None,
) -> SyntheticStream:
    low, high = content_range(vocab_size)
    synthetic = config["synthetic"]
    return SyntheticStream(
        source,
        synthetic["sources"][source],
        seed=seed,
        sequence_length=int(config["dataset"]["sequence_length"]),
        content_low=low,
        content_high=high,
        pad_id=0,
        mapping_seed=int(synthetic["byte_token_map_seed"]),
        bank=bank if source == "programs" else None,
    )


def heldout_batch(source: str, config: dict[str, Any], vocab_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """A fixed held-out synthetic set, disjoint in seed from every training stream."""
    synthetic = config["synthetic"]
    rows = int(synthetic["heldout_rows"])
    seed = int(synthetic["heldout_seed"])
    bank = None
    if source == "programs":
        length = int(config["dataset"]["sequence_length"]) + 1
        bank = _bank_chunk((synthetic["sources"]["programs"], seed, rows, length))
    return synthetic_stream(source, config, seed, vocab_size, bank).batch(rows)


def program_bank_path(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    key = {
        "params": config["synthetic"]["sources"]["programs"],
        "length": int(config["dataset"]["sequence_length"]) + 1,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return EXPERIMENT / "results" / "cache" / f"program-bank-{digest}.json", key


def load_program_bank(config: dict[str, Any], processes: int) -> tuple[list[list[int]], dict[str, Any]]:
    """Build the fixed program bank once and cache it; later runs reuse the cache."""
    path, key = program_bank_path(config)
    started = time.perf_counter()
    if not path.exists():
        rows = build_program_bank(key["params"], length=key["length"], processes=processes)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps({**key, "rows": rows}) + "\n", encoding="utf-8")
        partial.replace(path)
        built = True
    else:
        built = False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["params"] != key["params"] or payload["length"] != key["length"]:
        raise RuntimeError(f"cached program bank {path} does not match the config")
    return payload["rows"], {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "rows": len(payload["rows"]),
        "built_this_invocation": built,
        "seconds": time.perf_counter() - started,
    }


def synthetic_stage(
    model: nn.Module,
    source: str,
    *,
    seed: int,
    config: dict[str, Any],
    vocab_size: int,
    device: torch.device,
    bank: list[list[int]] | None,
) -> dict[str, Any]:
    synthetic = config["synthetic"]
    training = config["training"]
    steps = int(synthetic["steps"])
    stream = synthetic_stream(source, config, seed + int(synthetic["seed_offset"]), vocab_size, bank)
    heldout = heldout_batch(source, config, vocab_size)
    eval_steps = set(int(step) for step in synthetic["evaluation_steps"])
    batch_rows = int(config["evaluation"]["batch_size"])
    optimizer = new_optimizer(model, training)
    seconds = {"generation": 0.0, "training": 0.0, "evaluation": 0.0}
    curve = [{"step": 0, "heldout_nll": masked_nll(model, *heldout, device=device, batch=batch_rows)}]
    supervised = 0
    running: list[float] = []
    for step in range(steps):
        started = time.perf_counter()
        inputs, targets, mask = stream.batch(int(training["batch_size"]))
        seconds["generation"] += time.perf_counter() - started
        supervised += int(mask.sum())
        synchronize(device)
        started = time.perf_counter()
        running.append(
            train_step(
                model, optimizer, inputs, targets, mask,
                step=step, total_steps=steps, training=training, device=device,
            )
        )
        synchronize(device)
        seconds["training"] += time.perf_counter() - started
        if step + 1 in eval_steps:
            started = time.perf_counter()
            curve.append(
                {
                    "step": step + 1,
                    "heldout_nll": masked_nll(model, *heldout, device=device, batch=batch_rows),
                    "mean_training_loss_since_last": sum(running) / len(running),
                }
            )
            running = []
            seconds["evaluation"] += time.perf_counter() - started
    return {
        "source": source,
        "steps": steps,
        "heldout_curve": curve,
        "supervised_tokens": supervised,
        "seconds": seconds,
    }


def natural_stage(
    model: nn.Module,
    *,
    seed: int,
    steps: int,
    probe_steps: list[int],
    data: dict[str, Any],
    config: dict[str, Any],
    prompts: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    natural = config["natural"]
    interval = int(config["evaluation"]["interval"])
    batch_rows = int(config["evaluation"]["batch_size"])
    stream = NaturalStream(
        data["train"],
        sequence_length=int(config["dataset"]["sequence_length"]),
        seed=seed + int(natural["stream_seed_offset"]),
    )
    optimizer = new_optimizer(model, training)
    seconds = {"training": 0.0, "evaluation": 0.0}
    started = time.perf_counter()
    evaluations = [{"step": 0, "validation_nll": validation_nll(model, data["validation_windows"], device=device, batch=batch_rows)}]
    probes = {"0": evaluate_probes(model, prompts, device=device)}
    seconds["evaluation"] += time.perf_counter() - started
    nonfinite = False
    for step in range(steps):
        inputs, targets = stream.batch(int(training["batch_size"]))
        synchronize(device)
        started = time.perf_counter()
        loss = train_step(
            model, optimizer, inputs, targets, None,
            step=step, total_steps=steps, training=training, device=device,
        )
        synchronize(device)
        seconds["training"] += time.perf_counter() - started
        nonfinite = nonfinite or not math.isfinite(loss)
        if (step + 1) % interval == 0 or step + 1 == steps:
            started = time.perf_counter()
            nll = validation_nll(model, data["validation_windows"], device=device, batch=batch_rows)
            nonfinite = nonfinite or not math.isfinite(nll)
            evaluations.append({"step": step + 1, "validation_nll": nll})
            if step + 1 in probe_steps:
                probes[str(step + 1)] = evaluate_probes(model, prompts, device=device)
            seconds["evaluation"] += time.perf_counter() - started
    return {
        "steps": steps,
        "evaluations": evaluations,
        "probes": probes,
        "seconds": seconds,
        "nonfinite": nonfinite,
    }


def shuffle_within_tensors(model: nn.Module, seed: int) -> None:
    """Permute every parameter tensor's entries: same values, no learned structure."""
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for _, parameter in model.named_parameters():  # tied weights appear once
            flat = parameter.detach().flatten().cpu()
            permutation = torch.randperm(flat.numel(), generator=generator)
            parameter.copy_(flat[permutation].view_as(parameter).to(parameter.device))


def state_sha256(model: nn.Module) -> str:
    """SHA-256 over parameter names and float32 bytes (no NumPy dependency)."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        raw = parameter.detach().float().cpu().contiguous().flatten().view(torch.uint8)
        digest.update(bytes(raw.tolist()))
    return digest.hexdigest()


def warm_model(
    source: str,
    *,
    seed: int,
    config: dict[str, Any],
    vocab_size: int,
    device: torch.device,
    checkpoint_dir: Path,
    bank: list[list[int]] | None,
) -> tuple[nn.Module, dict[str, Any]]:
    """The post-synthetic model for (source, seed), from disk or by training it."""
    checkpoint = checkpoint_dir / f"{source}__seed{seed}.pt"
    model = new_model(config, seed, vocab_size, device)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        if state_sha256(model) != saved["summary"]["state_sha256"]:
            raise RuntimeError(f"checkpoint {checkpoint} does not match its recorded hash")
        return model, saved["summary"]
    summary = synthetic_stage(
        model, source, seed=seed, config=config, vocab_size=vocab_size, device=device, bank=bank
    )
    summary["state_sha256"] = state_sha256(model)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_suffix(".pt.partial")
    torch.save({"state_dict": model.state_dict(), "summary": summary}, partial)
    partial.replace(checkpoint)
    return model, summary


def run_arm(
    arm: str,
    *,
    seed: int,
    data: dict[str, Any],
    config: dict[str, Any],
    prompts: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    bank: list[list[int]] | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    vocab_size = len(data["vocabulary"].tokens)
    natural = config["natural"]
    primary = int(natural["warm_steps"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synthetic: dict[str, Any] | None = None
    if arm == "scratch":
        model = new_model(config, seed, vocab_size, device)
        steps = int(natural["scratch_steps"])
        probe_steps = [primary, steps]
    else:
        source = arm.removesuffix("_shuffled")
        model, synthetic = warm_model(
            source, seed=seed, config=config, vocab_size=vocab_size, device=device,
            checkpoint_dir=checkpoint_dir, bank=bank,
        )
        if arm.endswith("_shuffled"):
            shuffle_within_tensors(model, seed + int(config["shuffle_seed_offset"]))
        steps = primary
        probe_steps = [primary]
    initial_state = state_sha256(model)
    result = natural_stage(
        model, seed=seed, steps=steps, probe_steps=probe_steps, data=data,
        config=config, prompts=prompts, device=device,
    )
    return {
        "arm": arm,
        "seed": seed,
        "parameters": count_parameters(model),
        "synthetic_steps": 0 if synthetic is None else int(synthetic["steps"]),
        "natural_steps": steps,
        "natural_start_state_sha256": initial_state,
        "synthetic": synthetic,
        "natural": result,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
    }


def load_data(config: dict[str, Any]) -> dict[str, Any]:
    data = prepare_natural_data(config["dataset"])
    length = int(config["dataset"]["sequence_length"])
    data["validation_windows"] = validation_windows(data["validation"], length)
    data["unigram_validation_nll"] = unigram_nll(
        data["train"], data["validation_windows"], len(data["vocabulary"].tokens)
    )
    return data


def validate_config(config: dict[str, Any]) -> None:
    if list(config["arms"]) != list(ARMS):
        raise ValueError("config arms must be exactly the registered arms, in order")
    natural = config["natural"]
    expected = int(config["synthetic"]["steps"]) + int(natural["warm_steps"])
    if int(natural["scratch_steps"]) != expected:
        raise ValueError(f"scratch_steps must equal synthetic + warm natural steps ({expected})")
    if float(config["training"]["min_learning_rate_ratio"]) != 1.0:
        raise ValueError("E007 requires a constant post-warmup learning rate")
    interval = int(config["evaluation"]["interval"])
    if int(natural["warm_steps"]) % interval or int(natural["scratch_steps"]) % interval:
        raise ValueError("natural budgets must be multiples of the evaluation interval")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--bank-processes",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU processes for the one-time program-bank build (the bank itself does not depend on this)",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    configure_determinism()
    torch.set_float32_matmul_precision("high")
    data = load_data(config)
    if int(config["probes"]["token_high"]) > len(data["vocabulary"].tokens):
        raise ValueError("probe token range exceeds the vocabulary")
    prompts = probe_prompts(config["probes"])
    bank, bank_record = load_program_bank(config, max(1, int(args.bank_processes)))
    registered = config["synthetic"].get("registered_program_bank_sha256")
    bank_record["matches_registered"] = None if registered is None else bank_record["sha256"] == registered
    if bank_record["matches_registered"] is False:
        # Same distribution, different draw (e.g. a torch RNG change); recorded, not fatal.
        print(json.dumps({"event": "warning", "detail": "program bank differs from the registered build"}), flush=True)
    print(json.dumps({"event": "program_bank", **bank_record}), flush=True)
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
        "program_bank": {key: bank_record[key] for key in ("path", "sha256", "rows", "matches_registered")},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "training_autocast": (
                "bfloat16"
                if device.type == "cuda" and config["training"]["use_bfloat16"]
                else "float32"
            ),
            "evaluation_precision": "float32",
        },
    }
    runs = [(arm, int(seed)) for seed in config["seeds"] for arm in config["arms"]]
    for arm, seed in runs:
        target = output_dir / f"{arm}__seed{seed}.json"
        if target.exists():
            print(json.dumps({"event": "skip_existing", "file": target.name}), flush=True)
            continue
        result = run_arm(
            arm, seed=seed, data=data, config=config, prompts=prompts, device=device,
            checkpoint_dir=output_dir / "checkpoints", bank=bank,
        )
        payload = {**header, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "run": result}
        partial = target.with_suffix(".json.partial")
        partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        partial.replace(target)
        final = result["natural"]["evaluations"][-1]
        print(json.dumps({"event": "run_complete", "file": target.name, "final_step": final["step"]}), flush=True)
    print(json.dumps({"event": "complete", "runs": len(runs)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
