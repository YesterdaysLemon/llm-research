from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
E002_RESULT = (
    ROOT
    / "experiments"
    / "E002-contextual-transition-executor"
    / "results"
    / "confirm-affine.json"
)
E002_CONFIG = (
    ROOT
    / "experiments"
    / "E002-contextual-transition-executor"
    / "config"
    / "confirm-affine.json"
)
E003_DIR = ROOT / "experiments" / "E003-label-geometry-specificity"
E003_RESULT = E003_DIR / "results" / "confirm-label-geometry.json"
E003_CONFIG = E003_DIR / "config" / "confirm-label-geometry.json"
OUTPUT = E003_DIR / "results" / "analysis.json"
CHANCE = 1 / 47


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_rows(payload: dict[str, Any], condition_id: str) -> list[dict[str, Any]]:
    rows = [row for row in payload["runs"] if row["condition_id"] == condition_id]
    return sorted(rows, key=lambda row: int(row["seed"]))


def sample(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "condition_id": rows[0]["condition_id"],
        "id_accuracy": sample([float(row["id_accuracy"]) for row in rows]),
        "evaluation_accuracy": sample(
            [float(row["evaluation_accuracy"]) for row in rows]
        ),
        "train_seconds": sample([float(row["train_seconds"]) for row in rows]),
    }
    output["id_depth"] = {
        depth: sample(
            [float(row["id_accuracy_by_depth"][depth]) for row in rows]
        )
        for depth in ("2", "3", "4")
    }
    output["evaluation_depth"] = {
        depth: sample(
            [float(row["evaluation_accuracy_by_depth"][depth]) for row in rows]
        )
        for depth in ("2", "3", "4")
    }
    output["normalized_transfer"] = {
        depth: (
            (output["evaluation_depth"][depth]["mean"] - CHANCE)
            / (output["id_depth"][depth]["mean"] - CHANCE)
            if output["id_depth"][depth]["mean"] > CHANCE
            else None
        )
        for depth in ("2", "3", "4")
    }
    return output


def paired(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    depth: str | None = None,
) -> dict[str, Any]:
    left_seeds = [int(row["seed"]) for row in left_rows]
    right_seeds = [int(row["seed"]) for row in right_rows]
    if left_seeds != right_seeds:
        raise ValueError(f"paired seeds do not align: {left_seeds} != {right_seeds}")
    if depth is None:
        differences = np.asarray(
            [
                float(left["evaluation_accuracy"])
                - float(right["evaluation_accuracy"])
                for left, right in zip(left_rows, right_rows, strict=True)
            ],
            dtype=np.float64,
        )
    else:
        differences = np.asarray(
            [
                float(left["evaluation_accuracy_by_depth"][depth])
                - float(right["evaluation_accuracy_by_depth"][depth])
                for left, right in zip(left_rows, right_rows, strict=True)
            ],
            dtype=np.float64,
        )
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    critical = float(stats.t.ppf(0.975, differences.size - 1))
    half_width = critical * sd / math.sqrt(differences.size)
    return {
        "depth": depth,
        "n": int(differences.size),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "per_seed": differences.tolist(),
    }


def main() -> None:
    e002 = load(E002_RESULT)
    e003 = load(E003_RESULT)
    e002_config = load(E002_CONFIG)
    e003_config = load(E003_CONFIG)
    for key in ("data", "teacher", "training", "student_seeds"):
        if e002_config[key] != e003_config[key]:
            raise ValueError(f"E003 does not preserve E002 field: {key}")
    if e003["git_status_porcelain"]:
        raise ValueError("E003 run did not begin from a clean worktree")
    if e003["config_sha256"] != "a78736731718cbe4c610d75e0474261ea768d6df5c6310187307759761ed944c":
        raise ValueError("unexpected E003 configuration hash")

    label_rows = condition_rows(e003, "transformer-label-geometry")
    relational_rows = condition_rows(e002, "transformer-relational")
    logits_rows = condition_rows(e002, "transformer-logits")
    if len(label_rows) != 10:
        raise ValueError(f"expected ten E003 rows, found {len(label_rows)}")

    primary = {
        "overall": paired(relational_rows, label_rows),
        **{
            f"depth{depth}": paired(relational_rows, label_rows, depth=depth)
            for depth in ("2", "3", "4")
        },
    }
    secondary = {
        "overall": paired(label_rows, logits_rows),
        **{
            f"depth{depth}": paired(label_rows, logits_rows, depth=depth)
            for depth in ("2", "3", "4")
        },
    }
    if primary["overall"]["ci95_low"] > 0:
        decision = "learned_teacher_advantage_beyond_terminal_label_equivalence"
    elif primary["overall"]["ci95_high"] < 0:
        decision = "target_label_geometry_outperformed_learned_teacher_geometry"
    else:
        decision = "teacher_specificity_inconclusive"

    result = {
        "experiment_id": e003["experiment_id"],
        "config_sha256": e003["config_sha256"],
        "run_git_commit": e003["git_commit"],
        "run_git_status_porcelain": e003["git_status_porcelain"],
        "teacher_gate": {
            "overall": e003["teacher"]["evaluation"]["accuracy"],
            "by_depth": e003["teacher"]["evaluation"]["accuracy_by_depth"],
            "passed": (
                e003["teacher"]["evaluation"]["accuracy"] >= 0.95
                and min(e003["teacher"]["evaluation"]["accuracy_by_depth"].values())
                >= 0.95
            ),
        },
        "label_geometry": summarize(label_rows),
        "primary_relational_minus_label_geometry": primary,
        "secondary_label_geometry_minus_logits": secondary,
        "registered_decision": decision,
        "wall_time_comparability": (
            "invalid for cross-run efficiency comparison because host contention caused "
            "large seed-level runtime variation"
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
