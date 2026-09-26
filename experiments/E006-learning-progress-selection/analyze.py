"""Apply E006's preregistered decision rules to the raw fixed-run files.

Usage:
    python experiments/E006-learning-progress-selection/analyze.py \
        [--results results/fixed] [--config config/fixed.json] [--output results/analysis.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

EXPERIMENT = Path(__file__).resolve().parent

# Two-sided 95% Student-t critical values, used when SciPy is unavailable.
T_975 = {
    1: 12.706204736174698,
    2: 4.302652729749464,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.5705818356363146,
    6: 2.4469118511449692,
    7: 2.3646242515927844,
    8: 2.306004135204166,
    9: 2.2621571627982053,
}

PRIMARY = {
    "C1_absolute_minus_uniform": ("absolute", "uniform"),
    "C2_absolute_minus_loss": ("absolute", "loss"),
    "C3_absolute_minus_gradnorm": ("absolute", "gradnorm"),
}
SECONDARY = {
    "cosine_minus_gradnorm": ("cosine", "gradnorm"),
    "signed_minus_absolute": ("signed", "absolute"),
    "loss_minus_uniform": ("loss", "uniform"),
    "gradnorm_minus_uniform": ("gradnorm", "uniform"),
    "cosine_minus_uniform": ("cosine", "uniform"),
    "signed_minus_uniform": ("signed", "uniform"),
}


def t_critical(df: int) -> float:
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, df))
    except ImportError:
        return T_975[df]


def paired_interval(differences: list[float]) -> dict[str, Any]:
    n = len(differences)
    center = mean(differences)
    spread = stdev(differences) if n > 1 else float("nan")
    half = t_critical(n - 1) * spread / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": center,
        "sd": spread,
        "ci95": [center - half, center + half],
        "differences": differences,
    }


def nll_at(run: dict[str, Any], step: int) -> float:
    for row in run["evaluations"]:
        if int(row["step"]) == int(step):
            return float(row["validation_nll"])
    raise KeyError(f"no evaluation at step {step}")


def load_runs(results: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    runs = {}
    for path in sorted(results.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = payload["run"]
        runs[(run["condition"], run["arm"], int(run["seed"]))] = payload
    return runs


def check_provenance(payloads: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    expected = {
        (condition["id"], arm, int(seed))
        for condition in config["conditions"]
        for arm in config["arms"]
        for seed in config["seeds"]
    }
    present = set(payloads)
    single = lambda key: sorted({json.dumps(p[key], sort_keys=True) for p in payloads.values()})  # noqa: E731
    return {
        "expected_runs": len(expected),
        "present_runs": len(present & expected),
        "missing": sorted("/".join(map(str, key)) for key in expected - present),
        "unexpected": sorted("/".join(map(str, key)) for key in present - expected),
        "config_sha256": single("config_sha256"),
        "git_commit": single("git_commit"),
        "dataset": single("dataset"),
        "consistent": len(single("config_sha256")) == 1
        and len(single("git_commit")) == 1
        and len(single("dataset")) == 1,
    }


def gates(payloads: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rules = config["gates"]
    step = int(config["analysis"]["primary_step"])
    capability = {}
    for seed in config["seeds"]:
        payload = payloads[("clean", "uniform", int(seed))]
        improvement = payload["dataset"]["unigram_validation_nll"] - nll_at(payload["run"], step)
        capability[str(seed)] = improvement
    worst_error = 0.0
    for payload in payloads.values():
        for row in payload["run"]["diagnostics"]:
            scale = max(float(row["jvp_max_abs"]), 1e-12)
            worst_error = max(worst_error, float(row["jvp_vs_per_sample_max_abs_error"]) / scale)
    finite = all(not payload["run"]["nonfinite"] for payload in payloads.values())
    checks = {
        "uniform_beats_unigram": min(capability.values())
        >= float(rules["uniform_clean_nll_improvement_over_unigram_min"]),
        "jvp_matches_per_sample_gradients": worst_error
        <= float(rules["jvp_consistency_relative_tolerance"]),
        "all_runs_finite": finite,
    }
    return {
        **checks,
        "all": all(checks.values()),
        "uniform_improvement_over_unigram_by_seed": capability,
        "worst_relative_jvp_error": worst_error,
    }


def contrast(
    payloads: dict[tuple[str, str, int], dict[str, Any]],
    seeds: list[int],
    condition: str,
    left: tuple[str, int],
    right: tuple[str, int],
) -> dict[str, Any]:
    differences = [
        nll_at(payloads[(condition, left[0], seed)]["run"], left[1])
        - nll_at(payloads[(condition, right[0], seed)]["run"], right[1])
        for seed in seeds
    ]
    return paired_interval(differences)


def area_under_curve(run: dict[str, Any], steps: list[int]) -> float:
    points = [(step, nll_at(run, step)) for step in steps]
    area = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:]))
    return area / (points[-1][0] - points[0][0])


def decide(primary: dict[str, dict[str, Any]], gate_result: dict[str, Any]) -> str:
    if not gate_result["all"]:
        return "invalid: a validity gate failed; no H-013 inference"
    below = lambda name: primary[name]["ci95"][1] < 0  # noqa: E731
    signal = all(below(name) for name in PRIMARY)
    if signal and below("C4_absolute_minus_uniform_flops_matched"):
        return "supported: absolute learning-progress selection beats uniform, loss, and gradient-norm selection at matched steps and uniform training at matched FLOPs"
    if signal:
        return "selection signal supported; compute efficiency not supported (C4 interval does not exclude zero or favors uniform)"
    worse = [name for name in PRIMARY if primary[name]["ci95"][0] > 0]
    suffix = f"; absolute is reliably worse on {', '.join(worse)}" if worse else ""
    return "not supported: absolute selection fails to beat every matched-step comparator" + suffix


def analyze(payloads: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    seeds = [int(seed) for seed in config["seeds"]]
    step = int(config["analysis"]["primary_step"])
    flops_step = int(config["analysis"]["flops_matched_uniform_step"])
    shared_steps = sorted(int(value) for value in config["evaluation"]["steps"])
    output: dict[str, Any] = {"provenance": check_provenance(payloads, config)}
    if output["provenance"]["missing"]:
        output["decision"] = "incomplete: missing registered runs"
        return output
    gate_result = gates(payloads, config)
    output["gates"] = gate_result
    for condition in [c["id"] for c in config["conditions"]]:
        block: dict[str, Any] = {}
        primary = {
            name: contrast(payloads, seeds, condition, (left, step), (right, step))
            for name, (left, right) in PRIMARY.items()
        }
        primary["C4_absolute_minus_uniform_flops_matched"] = contrast(
            payloads, seeds, condition, ("absolute", step), ("uniform", flops_step)
        )
        block["primary_contrasts"] = primary
        block["secondary_contrasts"] = {
            name: contrast(payloads, seeds, condition, (left, step), (right, step))
            for name, (left, right) in SECONDARY.items()
        }
        block["nll_at_primary_step"] = {
            arm: [nll_at(payloads[(condition, arm, seed)]["run"], step) for seed in seeds]
            for arm in config["arms"]
        }
        block["uniform_nll_at_flops_matched_step"] = [
            nll_at(payloads[(condition, "uniform", seed)]["run"], flops_step) for seed in seeds
        ]
        block["area_under_nll_curve"] = {
            arm: mean(area_under_curve(payloads[(condition, arm, seed)]["run"], shared_steps) for seed in seeds)
            for arm in config["arms"]
        }
        shares = {}
        seconds = {}
        for arm in config["arms"]:
            per_seed = []
            timing = []
            for seed in seeds:
                run = payloads[(condition, arm, seed)]["run"]
                row = next(r for r in run["evaluations"] if int(r["step"]) == step)
                per_seed.append(row["noise_windows_selected"] / row["windows_selected"])
                timing.append(row["method_seconds"])
            shares[arm] = {"mean": mean(per_seed), "by_seed": per_seed}
            seconds[arm] = mean(timing)
        block["noise_share_selected_at_primary_step"] = shares
        block["mean_method_seconds_at_primary_step"] = seconds
        correlations: dict[str, list[float]] = {}
        for seed in seeds:
            for arm in config["arms"]:
                for row in payloads[(condition, arm, seed)]["run"]["diagnostics"]:
                    if int(row["step"]) > step:
                        continue
                    for pair, value in row["spearman"].items():
                        if value == value:  # skip NaN
                            correlations.setdefault(pair, []).append(float(value))
        block["mean_spearman_between_scores"] = {pair: mean(values) for pair, values in correlations.items()}
        if condition == "clean":
            output["decision"] = decide(primary, gate_result)
        else:
            fraction = next(c["noise_fraction"] for c in config["conditions"] if c["id"] == condition)
            block["registered_predictions"] = {
                "P1_loss_selects_mostly_noise": shares["loss"]["mean"] > 0.5,
                "P2_absolute_overselects_noise": shares["absolute"]["mean"] > fraction,
                "P3_signed_underselects_noise": shares["signed"]["mean"] < fraction,
                "P4_signed_beats_absolute": block["secondary_contrasts"]["signed_minus_absolute"]["ci95"][1] < 0,
            }
        output[condition] = block
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=EXPERIMENT / "results" / "fixed")
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config" / "fixed.json")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results" / "analysis.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = analyze(load_runs(args.results), config)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result.get("decision")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
