from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(__file__).resolve().parent
GENERATED = PAPER_DIR / "generated"
GENERATED.mkdir(parents=True, exist_ok=True)


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


P0 = load("experiments/E001-trajectory-sufficiency/results/pilot.json")
P1_BUDGET = load(
    "experiments/E001-trajectory-sufficiency/results/p1-dev-budget.json"
)
P1_NARROW = load(
    "experiments/E001-trajectory-sufficiency/results/p1-dev-arch-4x128.json"
)
P1_WIDE = load(
    "experiments/E001-trajectory-sufficiency/results/p1-dev-arch-4x256.json"
)
AFFINE = load(
    "experiments/E002-contextual-transition-executor/results/confirm-affine.json"
)
BITWISE = load(
    "experiments/E002-contextual-transition-executor/results/confirm-bitwise.json"
)


def runs_by_condition(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in payload["runs"]:
        output[str(run["condition_id"])].append(run)
    return {key: sorted(value, key=lambda row: int(row["seed"])) for key, value in output.items()}


def sample_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def condition_summary(payload: dict[str, Any], condition_id: str) -> dict[str, Any]:
    rows = runs_by_condition(payload)[condition_id]
    output: dict[str, Any] = {
        "condition_id": condition_id,
        "stored_parameters": int(rows[0]["stored_parameters"]),
        "active_parameters_per_step": int(rows[0]["active_parameters_per_step"]),
        "accuracy": sample_summary([float(row["evaluation_accuracy"]) for row in rows]),
        "train_seconds": sample_summary([float(row["train_seconds"]) for row in rows]),
    }
    output["depth"] = {
        depth: sample_summary(
            [float(row["evaluation_accuracy_by_depth"][depth]) for row in rows]
        )
        for depth in ("2", "3", "4")
    }
    return output


def paired_difference(
    payload: dict[str, Any],
    left: str,
    right: str,
    *,
    depth: str | None = None,
) -> dict[str, Any]:
    grouped = runs_by_condition(payload)
    left_rows = grouped[left]
    right_rows = grouped[right]
    if [row["seed"] for row in left_rows] != [row["seed"] for row in right_rows]:
        raise ValueError("paired seeds do not align")
    if depth is None:
        differences = np.asarray(
            [
                float(a["evaluation_accuracy"]) - float(b["evaluation_accuracy"])
                for a, b in zip(left_rows, right_rows, strict=True)
            ]
        )
    else:
        differences = np.asarray(
            [
                float(a["evaluation_accuracy_by_depth"][depth])
                - float(b["evaluation_accuracy_by_depth"][depth])
                for a, b in zip(left_rows, right_rows, strict=True)
            ]
        )
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    critical = float(stats.t.ppf(0.975, differences.size - 1))
    half_width = critical * sd / math.sqrt(differences.size)
    return {
        "left": left,
        "right": right,
        "depth": depth,
        "n": int(differences.size),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "per_seed": differences.tolist(),
    }


CONDITIONS = [
    "table-labels",
    "gru-labels",
    "transformer-labels",
    "transformer-logits",
    "transformer-relational",
    "transformer-shuffled",
    "transformer-random",
    "transformer-untrained",
]


def family_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    summaries = {name: condition_summary(payload, name) for name in CONDITIONS}
    paired = {}
    for baseline in ("transformer-logits", "transformer-shuffled"):
        paired[f"relational_minus_{baseline.removeprefix('transformer-')}_overall"] = paired_difference(
            payload, "transformer-relational", baseline
        )
        for depth in ("3", "4"):
            paired[
                f"relational_minus_{baseline.removeprefix('transformer-')}_depth{depth}"
            ] = paired_difference(
                payload, "transformer-relational", baseline, depth=depth
            )
    table = summaries["table-labels"]
    relational = summaries["transformer-relational"]
    efficiency = {
        "stored_parameter_ratio_transformer_to_table": relational[
            "stored_parameters"
        ]
        / table["stored_parameters"],
        "active_parameter_ratio_transformer_to_table": relational[
            "active_parameters_per_step"
        ]
        / table["active_parameters_per_step"],
        "training_time_ratio_relational_to_table": relational["train_seconds"][
            "mean"
        ]
        / table["train_seconds"]["mean"],
        "accuracy_per_parameter_ratio_table_to_relational": (
            table["accuracy"]["mean"] / table["stored_parameters"]
        )
        / (relational["accuracy"]["mean"] / relational["stored_parameters"]),
    }
    return {"conditions": summaries, "paired": paired, "efficiency": efficiency}


analysis = {
    "affine": family_analysis(AFFINE),
    "bitwise": family_analysis(BITWISE),
    "teacher_gates": {
        "affine": {
            "overall": AFFINE["teacher"]["evaluation"]["accuracy"],
            "depth4": AFFINE["teacher"]["evaluation"]["accuracy_by_depth"]["4"],
            "passed_registered_95_percent_floor": (
                AFFINE["teacher"]["evaluation"]["accuracy"] >= 0.95
                and min(AFFINE["teacher"]["evaluation"]["accuracy_by_depth"].values())
                >= 0.95
            ),
        },
        "bitwise": {
            "overall": BITWISE["teacher"]["evaluation"]["accuracy"],
            "depth4": BITWISE["teacher"]["evaluation"]["accuracy_by_depth"]["4"],
            "passed_registered_95_percent_floor": (
                BITWISE["teacher"]["evaluation"]["accuracy"] >= 0.95
                and min(BITWISE["teacher"]["evaluation"]["accuracy_by_depth"].values())
                >= 0.95
            ),
        },
    },
    "chance": {"affine": 1 / 47, "bitwise": 1 / 64},
    "config_sha256": {
        "affine": AFFINE["config_sha256"],
        "bitwise": BITWISE["config_sha256"],
    },
}

(GENERATED / "statistics.json").write_text(
    json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
)


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


lines = [
    "# Generated result tables",
    "",
    "Generated by `paper/analyze.py` from committed raw JSON.",
]
for family in ("affine", "bitwise"):
    lines.extend(
        [
            "",
            f"## {family.title()}",
            "",
            "| Condition | Parameters | Accuracy | Depth 2 | Depth 3 | Depth 4 | Train time |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in CONDITIONS:
        item = analysis[family]["conditions"][name]
        lines.append(
            f"| {name} | {item['stored_parameters']:,} | "
            f"{percent(item['accuracy']['mean'])} +/- {percent(item['accuracy']['sd'])} | "
            f"{percent(item['depth']['2']['mean'])} | {percent(item['depth']['3']['mean'])} | "
            f"{percent(item['depth']['4']['mean'])} | {item['train_seconds']['mean']:.2f} s |"
        )
    lines.extend(["", "Paired relational effects:", ""])
    for key, item in analysis[family]["paired"].items():
        lines.append(
            f"- {key}: {percent(item['mean'])} "
            f"[{percent(item['ci95_low'])}, {percent(item['ci95_high'])}]"
        )
(GENERATED / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


COLORS = {
    "table-labels": "#1b9e77",
    "gru-labels": "#7570b3",
    "transformer-logits": "#666666",
    "transformer-relational": "#d95f02",
    "transformer-shuffled": "#e6ab02",
}
LABELS = {
    "table-labels": "Transition table",
    "gru-labels": "GRU",
    "transformer-logits": "Transformer + logits",
    "transformer-relational": "Transformer + relational",
    "transformer-shuffled": "Transformer + shuffled",
}


def style_axes(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2, linewidth=0.7)


selected = list(COLORS)
depths = ("2", "3", "4")
x = np.arange(len(depths))
width = 0.16
fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
for index, name in enumerate(selected):
    item = analysis["affine"]["conditions"][name]
    values = [100 * item["depth"][depth]["mean"] for depth in depths]
    axis.bar(
        x + (index - 2) * width,
        values,
        width,
        label=LABELS[name],
        color=COLORS[name],
    )
axis.axhline(100 / 47, color="black", linestyle=":", linewidth=1, label="Chance")
axis.set_xticks(x, [f"Depth {depth}" for depth in depths])
axis.set_ylabel("Held-out pair accuracy (%)")
axis.set_ylim(0, 105)
axis.set_title("Affine confirmation: relational gains decay with composition depth")
style_axes(axis)
axis.legend(ncol=3, frameon=False, fontsize=8.5)
fig.savefig(GENERATED / "figure1_affine_depth.png", dpi=220)
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
for axis, family in zip(axes, ("affine", "bitwise"), strict=True):
    for name in (
        "table-labels",
        "gru-labels",
        "transformer-logits",
        "transformer-relational",
    ):
        item = analysis[family]["conditions"][name]
        axis.scatter(
            item["stored_parameters"],
            100 * item["accuracy"]["mean"],
            s=80,
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.annotate(
            LABELS[name],
            (item["stored_parameters"], 100 * item["accuracy"]["mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.8,
        )
    axis.set_xscale("log")
    axis.set_xlim(7_000, 2_000_000)
    axis.set_ylim(0, 105)
    axis.set_title(family.title())
    axis.set_xlabel("Stored parameters (log scale)")
    style_axes(axis)
axes[0].set_ylabel("Held-out pair accuracy (%)")
fig.suptitle("Task factorization changes the capability-per-parameter frontier")
fig.savefig(GENERATED / "figure2_efficiency_frontier.png", dpi=220)
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), constrained_layout=True)
for axis, family, payload in zip(
    axes, ("affine", "bitwise"), (AFFINE, BITWISE), strict=True
):
    grouped = runs_by_condition(payload)
    logits = grouped["transformer-logits"]
    relational = grouped["transformer-relational"]
    for left, right in zip(logits, relational, strict=True):
        axis.plot(
            [0, 1],
            [100 * left["evaluation_accuracy"], 100 * right["evaluation_accuracy"]],
            color="#777777",
            alpha=0.75,
            marker="o",
            markersize=4,
        )
    axis.set_xticks([0, 1], ["Logits", "Relational"])
    axis.set_xlim(-0.25, 1.25)
    axis.set_ylabel("Held-out pair accuracy (%)")
    suffix = "" if family == "affine" else " (descriptive; gate failed)"
    axis.set_title(f"{family.title()}{suffix}")
    style_axes(axis)
fig.suptitle("Relational geometry improves every paired student seed")
fig.savefig(GENERATED / "figure3_paired_effects.png", dpi=220)
plt.close(fig)


p1_points = [
    ("2x256\n24 ep", 1_082_927, P1_BUDGET["summary"][0]),
    ("2x256\n48 ep", 1_082_927, P1_BUDGET["summary"][1]),
    ("2x256\n72 ep", 1_082_927, P1_BUDGET["summary"][2]),
    ("4x128\n72 ep", 807_471, P1_NARROW["summary"][0]),
    ("4x256\n72 ep", 2_137_135, P1_WIDE["summary"][0]),
]
fig, axis = plt.subplots(figsize=(8.4, 4.5), constrained_layout=True)
locations = np.arange(len(p1_points))
id_values = [100 * point[2]["id_accuracy_mean"] for point in p1_points]
dev_values = [100 * point[2]["evaluation_accuracy_mean"] for point in p1_points]
axis.bar(locations - 0.18, id_values, 0.36, label="In distribution", color="#4c78a8")
axis.bar(locations + 0.18, dev_values, 0.36, label="Development pairs", color="#f58518")
axis.axhline(80, color="black", linestyle="--", linewidth=1, label="Capability gate")
axis.set_xticks(locations, [point[0] for point in p1_points])
axis.set_ylabel("Accuracy (%)")
axis.set_ylim(0, 105)
axis.set_title("E001-P1: larger and deeper students did not clear the capability gate")
style_axes(axis)
axis.legend(frameon=False, ncol=3, fontsize=8.5)
fig.savefig(GENERATED / "figure4_p1_gate.png", dpi=220)
plt.close(fig)

print(f"wrote analysis artifacts to {GENERATED}")
