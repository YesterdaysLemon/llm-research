"""Apply E007's preregistered decision rules to the raw fixed-run files.

Usage:
    python experiments/E007-procedural-pre-pretraining/analyze.py \
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
SOURCES = ("pcfg", "programs")

# Two-sided Student-t critical values for 95% and 97.5% intervals, used without SciPy.
T_TABLE = {
    0.95: {1: 12.706204736174698, 2: 4.302652729749464, 3: 3.182446305284263, 4: 2.7764451051977987},
    0.975: {1: 25.451699579357, 2: 6.205346816834, 3: 4.176535019296, 4: 3.495405708154},
}


def t_critical(confidence: float, df: int) -> float:
    try:
        from scipy import stats

        return float(stats.t.ppf(1 - (1 - confidence) / 2, df))
    except ImportError:
        return T_TABLE[confidence][df]


def paired_interval(differences: list[float], confidence: float) -> dict[str, Any]:
    n = len(differences)
    center = mean(differences)
    spread = stdev(differences) if n > 1 else float("nan")
    half = t_critical(confidence, n - 1) * spread / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": center,
        "sd": spread,
        "confidence": confidence,
        "ci": [center - half, center + half],
        "differences": differences,
    }


def curve(payload: dict[str, Any]) -> list[tuple[int, float]]:
    return [(int(row["step"]), float(row["validation_nll"])) for row in payload["run"]["natural"]["evaluations"]]


def nll_at(payload: dict[str, Any], step: int) -> float:
    for at, value in curve(payload):
        if at == int(step):
            return value
    raise KeyError(f"no natural evaluation at step {step}")


def steps_to_reach(points: list[tuple[int, float]], target: float) -> float | None:
    """First natural step at which the curve reaches `target`, linearly interpolated."""
    previous = None
    for step, value in points:
        if value <= target:
            if previous is None:
                return float(step)
            (left_step, left_value) = previous
            fraction = (left_value - target) / (left_value - value)
            return left_step + fraction * (step - left_step)
        previous = (step, value)
    return None


def load_runs(results: Path) -> dict[tuple[str, int], dict[str, Any]]:
    runs = {}
    for path in sorted(results.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[(payload["run"]["arm"], int(payload["run"]["seed"]))] = payload
    return runs


def provenance(payloads: dict[tuple[str, int], dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    expected = {(arm, int(seed)) for arm in config["arms"] for seed in config["seeds"]}
    distinct = lambda key: sorted({json.dumps(p.get(key), sort_keys=True) for p in payloads.values()})  # noqa: E731
    return {
        "expected_runs": len(expected),
        "present_runs": len(set(payloads) & expected),
        "missing": sorted(f"{arm}/{seed}" for arm, seed in expected - set(payloads)),
        "consistent": all(len(distinct(key)) <= 1 for key in ("config_sha256", "git_commit", "dataset", "program_bank")),
        "config_sha256": distinct("config_sha256"),
        "git_commit": distinct("git_commit"),
    }


def gates(payloads: dict[tuple[str, int], dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rules = config["gates"]
    primary = int(config["analysis"]["primary_natural_step"])
    seeds = [int(seed) for seed in config["seeds"]]
    capability = {
        str(seed): payloads[("scratch", seed)]["dataset"]["unigram_validation_nll"]
        - nll_at(payloads[("scratch", seed)], primary)
        for seed in seeds
    }
    learned: dict[str, dict[str, float]] = {}
    for source in SOURCES:
        learned[source] = {}
        for seed in seeds:
            heldout = payloads[(source, seed)]["run"]["synthetic"]["heldout_curve"]
            learned[source][str(seed)] = heldout[0]["heldout_nll"] - heldout[-1]["heldout_nll"]
    finite = all(not payload["run"]["natural"]["nonfinite"] for payload in payloads.values())
    minimum = float(rules["synthetic_heldout_improvement_min"])
    return {
        "scratch_beats_unigram": min(capability.values()) >= float(rules["scratch_nll_improvement_over_unigram_min"]),
        "all_runs_finite": finite,
        "synthetic_stage_learned": {source: min(values.values()) >= minimum for source, values in learned.items()},
        "scratch_improvement_over_unigram_by_seed": capability,
        "synthetic_heldout_improvement_by_seed": learned,
    }


def decide(per_source: dict[str, dict[str, Any]], gate_result: dict[str, Any]) -> str:
    if not (gate_result["scratch_beats_unigram"] and gate_result["all_runs_finite"]):
        return "invalid: a global validity gate failed; no H-012 inference"
    valid = [s for s in SOURCES if gate_result["synthetic_stage_learned"][s]]
    if not valid:
        return "invalid: no synthetic stage passed its learning gate"
    below = {
        s: {name: per_source[s][name]["ci"][1] < 0 for name in ("D1_vs_scratch", "D2_vs_compute_matched", "D3_vs_shuffled")}
        for s in valid
    }
    full = [s for s in valid if all(below[s].values())]
    if full:
        return f"supported: {', '.join(full)} beats scratch, compute-matched scratch, and its shuffled control"
    head = [s for s in valid if below[s]["D1_vs_scratch"] and below[s]["D3_vs_shuffled"]]
    if head:
        return (
            f"head start supported for {', '.join(head)}; not compute-efficient at this synthetic budget "
            "(compute-matched scratch not beaten)"
        )
    statistics_only = [s for s in valid if below[s]["D1_vs_scratch"]]
    if statistics_only:
        return (
            f"not supported as procedural structure: {', '.join(statistics_only)} beats scratch "
            "but not its shuffled control"
        )
    return "not supported: no warm start beats scratch at matched natural tokens"


def analyze(payloads: dict[tuple[str, int], dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"provenance": provenance(payloads, config)}
    if output["provenance"]["missing"]:
        output["decision"] = "incomplete: missing registered runs"
        return output
    seeds = [int(seed) for seed in config["seeds"]]
    analysis = config["analysis"]
    primary = int(analysis["primary_natural_step"])
    matched = int(analysis["compute_matched_scratch_step"])
    confidence = float(analysis["confidence"])
    gate_result = gates(payloads, config)
    output["gates"] = gate_result

    def contrast(left: tuple[str, int], right: tuple[str, int], level: float) -> dict[str, Any]:
        return paired_interval(
            [nll_at(payloads[(left[0], s)], left[1]) - nll_at(payloads[(right[0], s)], right[1]) for s in seeds],
            level,
        )

    per_source: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        shuffled = f"{source}_shuffled"
        block: dict[str, Any] = {
            "D1_vs_scratch": contrast((source, primary), ("scratch", primary), confidence),
            "D2_vs_compute_matched": contrast((source, primary), ("scratch", matched), confidence),
            "D3_vs_shuffled": contrast((source, primary), (shuffled, primary), confidence),
        }
        block["early_contrasts_95"] = {
            str(step): {
                "vs_scratch": contrast((source, step), ("scratch", step), 0.95),
                "vs_shuffled": contrast((source, step), (shuffled, step), 0.95),
            }
            for step in analysis["savings_target_steps"]
            if int(step) < primary
        }
        per_source[source] = block
    output["primary"] = per_source
    output["decision"] = decide(per_source, gate_result)

    savings: dict[str, dict[str, Any]] = {}
    for arm in config["arms"]:
        if arm == "scratch":
            continue
        savings[arm] = {}
        for k in analysis["savings_target_steps"]:
            fractions = []
            for seed in seeds:
                reached = steps_to_reach(curve(payloads[(arm, seed)]), nll_at(payloads[("scratch", seed)], int(k)))
                fractions.append(None if reached is None else 1 - reached / int(k))
            reached_values = [value for value in fractions if value is not None]
            savings[arm][str(k)] = {
                "by_seed": fractions,
                "mean_where_reached": mean(reached_values) if reached_values else None,
                "not_reached": sum(value is None for value in fractions),
            }
    output["natural_token_savings_to_scratch_level"] = savings
    output["nll_by_arm"] = {
        arm: {
            "natural_step_0": mean(nll_at(payloads[(arm, s)], 0) for s in seeds),
            "primary_step": mean(nll_at(payloads[(arm, s)], primary) for s in seeds),
        }
        for arm in config["arms"]
    }
    output["nll_by_arm"]["scratch"]["compute_matched_step"] = mean(nll_at(payloads[("scratch", s)], matched) for s in seeds)
    output["probes"] = {
        arm: {
            step: {
                probe: mean(payloads[(arm, s)]["run"]["natural"]["probes"][step][probe] for s in seeds)
                for probe in payloads[(arm, seeds[0])]["run"]["natural"]["probes"][step]
            }
            for step in payloads[(arm, seeds[0])]["run"]["natural"]["probes"]
        }
        for arm in config["arms"]
    }
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
