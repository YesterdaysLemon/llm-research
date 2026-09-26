"""Re-derive the numbers cited in research/self-play-pretraining-2026-09-26.md.

Reads only the scored data that Cowsik et al. ship with their figures in
https://github.com/nourya-aliz/Solomonoff-Figures (arXiv:2609.30063). No model
is run; nothing here re-scores checkpoints. Standard library only.

Usage:
    git clone https://github.com/nourya-aliz/Solomonoff-Figures
    git -C Solomonoff-Figures checkout 4b6d46725d9c4f42bd5fa78debbd2a2c960a6614
    python research/self-play-pretraining/reanalyze.py Solomonoff-Figures \
        --out research/self-play-pretraining/reanalysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

PINNED_COMMIT = "4b6d46725d9c4f42bd5fa78debbd2a2c960a6614"
PROGRAMS_PER_ROUND_SELFPLAY = 1536  # figures/fig2 README
CONTEXT = 4096
TEXT_LIKE = ["dclm", "llm_compression_cc", "llm_compression_python", "kolmogorov_text", "metamath"]
BROAD = TEXT_LIKE + ["arithmetic", "audio_8bit", "cifar10_rgb_planar", "mutopia_melody_16th", "dna"]
BUDGETS = [1e15, 1e16, 1e17, 1e18]

# Table 5 of the paper (v1 PDF), used to check the repository's summary CSV.
PAPER_TABLE5 = {
    "dclm": {"none": 5.34, "uniform": 7.75, "signed": 5.06, "shuffle": 5.96},
    "metamath": {"none": 3.38, "uniform": 7.39, "signed": 3.47, "shuffle": 4.39},
    "aitdcc_b_c_source": {"none": 4.16, "uniform": 7.92, "signed": 4.10, "shuffle": 4.79},
    "dna": {"none": 2.29, "uniform": 3.02, "signed": 2.45, "shuffle": 2.50},
    "arithmetic": {"none": 0.22, "uniform": 7.94, "signed": 0.51, "shuffle": 0.73},
    "audio_8bit": {"none": 2.27, "uniform": 3.93, "signed": 2.45, "shuffle": 2.91},
    "audio_16bit": {"none": 5.02, "uniform": 6.08, "signed": 5.24, "shuffle": 5.79},
    "mutopia_melody_16th": {"none": 2.20, "uniform": 7.09, "signed": 2.21, "shuffle": 3.26},
    "cifar10_rgb_planar": {"none": 5.96, "uniform": 7.83, "signed": 6.14, "shuffle": 7.14},
}

# Table 2 of the paper (v1 PDF): per-corpus compute exponents b.
PAPER_TABLE2 = {
    "dclm": 0.123,
    "cifar10_rgb_hwc": 0.066,
    "cifar10_rgb_planar": 0.145,
    "audio_16bit": 0.141,
    "audio_8bit": 0.260,
    "mutopia_melody_16th": 0.249,
    "metamath": 0.129,
    "dna": 0.435,
    "aitdcc_b_c_source": 0.116,
    "llm_compression_python": 0.113,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def check_commit(root: Path) -> str | None:
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if head != PINNED_COMMIT:
        print(f"warning: {root} is at {head}, not the pinned {PINNED_COMMIT}", file=sys.stderr)
    return head


def gap_share(uniform: float, arm: float, canonical: float) -> float:
    """Fraction of the uniform-to-canonical improvement that `arm` recovers."""
    return (uniform - arm) / (uniform - canonical)


def shuffle_shares(fig: Path) -> dict:
    rows = read_csv(fig / "table5_reward_ablations" / "reward_frontier_summary.csv")
    repo = {}
    for row in rows:
        repo.setdefault(row["corpus"], {})[row["arm"]] = float(row["bpb"])
    out = {}
    for corpus, paper in PAPER_TABLE5.items():
        arms = repo[corpus]
        out[corpus] = {
            "paper_table5": paper,
            "paper_shuffle_share": round(gap_share(paper["uniform"], paper["shuffle"], paper["none"]), 3),
            "repo_summary": {arm: round(arms[arm], 3) for arm in ("none", "uniform", "signed", "shuffle")},
            "repo_shuffle_share": round(gap_share(arms["uniform"], arms["shuffle"], arms["none"]), 3),
        }
    return out


def shuffle_share_by_round(fig: Path, rung: str = "d128h2L4", k: str = "4") -> dict:
    """Shuffle share at intermediate rounds of the 1M rung, K=4 ensembles."""
    data = fig / "fig2_transfer_across_modalities" / "data"
    selfplay = read_csv(data / "selfplay_frontier_perk.csv")
    uniform = read_csv(data / "uniform_frontier_perk.csv")
    cache = fig / "table5_reward_ablations" / "cache" / "shuffle"

    def lookup(rows, corpus, rnd):
        for row in rows:
            if row["rung"] == rung and row["corpus"] == corpus and row["K"] == k and int(row["round"]) == rnd:
                return float(row["bpb"])
        return None

    out = {}
    for rnd in (1024, 2048, 4096):
        path = cache / f"round_{rnd:06d}.json"
        if not path.exists():
            continue
        shuffled = read_json(path)["bpb"]
        cell = {}
        for corpus in ("dclm", "metamath", "arithmetic", "audio_8bit", "cifar10_rgb_planar"):
            canon, unif = lookup(selfplay, corpus, rnd), lookup(uniform, corpus, rnd)
            shuf = shuffled.get(corpus, {}).get(k)
            if None in (canon, unif, shuf):
                continue
            cell[corpus] = {
                "none": round(canon, 3),
                "uniform": round(unif, 3),
                "shuffle": round(shuf, 3),
                "shuffle_share": round(gap_share(unif, shuf, canon), 3),
            }
        out[str(rnd)] = cell
    return out


def matched_compute(fig: Path) -> dict:
    data = fig / "fig2_transfer_across_modalities" / "data"
    arms = {name: read_csv(data / f"{name}_frontier_perk.csv") for name in ("selfplay", "pcfg", "uniform")}
    per_round = {
        "pcfg": read_json(data / "pcfg_programs_per_round.json")["programs_per_round"],
        "uniform": read_json(data / "prior_programs_per_round.json")["programs_per_round"],
    }

    def compute(arm, row):
        programs = PROGRAMS_PER_ROUND_SELFPLAY if arm == "selfplay" else per_round[arm][row["rung"]]
        return int(row["K"]) * float(row["N"]) * programs * CONTEXT * (int(row["round"]) + 1)

    out = {}
    for corpus in BROAD:
        out[corpus] = {}
        for budget in BUDGETS:
            cell = {}
            for arm, rows in arms.items():
                values = [float(r["bpb"]) for r in rows if r["corpus"] == corpus and compute(arm, r) <= budget]
                cell[arm] = round(min(values), 3) if values else None
            out[corpus][f"{budget:.0e}"] = cell
    return out


def frontier_end(fig: Path) -> dict:
    rows = read_csv(fig / "fig2_transfer_across_modalities" / "data" / "selfplay_frontier_perk.csv")

    def compute(row):
        return int(row["K"]) * float(row["N"]) * PROGRAMS_PER_ROUND_SELFPLAY * CONTEXT * (int(row["round"]) + 1)

    out = {}
    for corpus in BROAD:
        corpus_rows = sorted((r for r in rows if r["corpus"] == corpus), key=compute)
        best, last = float("inf"), None
        for row in corpus_rows:
            if float(row["bpb"]) < best:
                best, last = float(row["bpb"]), row
        out[corpus] = {
            "last_frontier_compute": float(f"{compute(last):.3g}"),
            "last_frontier_point": f"{last['rung']} round {last['round']} K={last['K']}",
            "best_bpb": round(best, 3),
            "max_compute_scored": float(f"{max(compute(r) for r in corpus_rows):.3g}"),
        }
    return out


def text_trajectory(fig: Path) -> dict:
    rows = read_csv(fig / "fig2_transfer_across_modalities" / "data" / "selfplay_frontier_perk.csv")
    out = {}
    for k in ("1", "4", "8"):
        traj = sorted(
            (int(r["round"]), float(r["bpb"]))
            for r in rows
            if r["corpus"] == "dclm" and r["rung"] == "d512h8L8" and r["K"] == k
        )
        best = min(traj, key=lambda item: item[1])
        after = [b for rnd, b in traj if rnd >= 2048]
        out[f"K{k}"] = {
            "round_768": round(dict(traj)[768], 3),
            "best": {"round": best[0], "bpb": round(best[1], 3)},
            "range_from_round_2048": [round(min(after), 3), round(max(after), 3)],
            "final": {"round": traj[-1][0], "bpb": round(traj[-1][1], 3)},
        }
    return out


def plateau_rounds(fig: Path, corpus: str = "dclm", k: str = "4", tolerance: float = 0.05) -> dict:
    """First round at which each rung comes within `tolerance` bpb of its own best."""
    rows = read_csv(fig / "fig2_transfer_across_modalities" / "data" / "selfplay_frontier_perk.csv")
    out = {}
    for rung in sorted({r["rung"] for r in rows}):
        traj = sorted((int(r["round"]), float(r["bpb"])) for r in rows if r["rung"] == rung and r["corpus"] == corpus and r["K"] == k)
        if not traj:
            continue
        best = min(b for _, b in traj)
        first = next(rnd for rnd, b in traj if b <= best + tolerance)
        out[rung] = {"best_bpb": round(best, 3), "first_round_within_tol": first, "last_round": traj[-1][0]}
    return out


def exponents(fig: Path) -> dict:
    """Shipped fits (saturating, pure power law, late slope) beside the paper's Table 2."""
    rows = read_csv(fig / "table2_scaling_exponents" / "scaling_exponents.csv")
    out = {}
    for r in rows:
        if r["corpus"] not in PAPER_TABLE2 and r["corpus"] != "__mean__":
            continue
        cell = {key: round(float(r[key]), 4) for key in ("alpha_sat", "alpha_pow", "alpha_late", "E", "best_bpb", "decades_C")}
        cell["paper_table2"] = PAPER_TABLE2.get(r["corpus"])
        out[r["corpus"]] = cell
    return out


def curriculum_value(fig: Path) -> dict:
    rows = read_csv(fig / "fig3_curriculum_value" / "data" / "plotted_values.csv")
    return {
        r["arm"]: {
            "epiplexity_mean": round(float(r["epiplexity_mean"]), 1),
            "dclm_subset_mean": round(float(r["dclm_subset_mean"]), 3),
            "audio_16bit_subset_mean": round(float(r["audio_16bit_subset_mean"]), 3),
            "cifar10_subset_mean": round(float(r["cifar10_rgb_planar_subset_mean"]), 3),
        }
        for r in rows
    }


def pre_pretraining(fig: Path) -> dict:
    data = read_json(fig / "fig6_pre_pretraining" / "fig6_data_cache.json")
    out = {}
    for panel, arms in data.items():
        cell = {}
        for arm, values in arms.items():
            tokens = [end[0] for end in values["ends"]]
            losses = [end[1] for end in values["ends"]]
            cell[arm] = {
                "initial_bpb_mean": round(mean(values["inits"]), 3),
                "final_bpb_mean": round(mean(losses), 4),
                "final_bpb_range": [round(min(losses), 4), round(max(losses), 4)],
                "tokens_to_plateau_M_mean": round(mean(tokens), 1),
            }
        scratch, warm = cell["scratch"], cell["warm"]
        cell["final_gap_bpb"] = round(scratch["final_bpb_mean"] - warm["final_bpb_mean"], 4)
        cell["token_saving_fraction"] = round(1 - warm["tokens_to_plateau_M_mean"] / scratch["tokens_to_plateau_M_mean"], 3)
        out[panel] = cell
    return out


def icl(fig: Path) -> dict:
    out = {}
    for arm in ("selfplay", "uniform_prior", "pcfg"):
        base = fig / "fig4_icl_across_methods" / "data" / arm
        best = {}
        for name in ("icl_results.json", "icl_v3_results.json", "icl_v4_results.json"):
            results = read_json(base / name)
            results = results.get("results", results)
            for key, value in results.items():
                if not isinstance(value, dict):
                    continue
                score = value.get("acc", value.get("acc_mean"))
                if score is None:
                    continue
                task = key.split("|")[0]
                best[task] = max(best.get(task, 0.0), score)
        out[arm] = {task: round(best[task], 3) for task in ("first", "last", "max", "min", "sum", "mean", "assoc", "palin", "stack") if task in best}
    return out


def discovered_structure(fig: Path) -> dict:
    counts: dict[str, int] = {}
    with (fig / "table1_discovered_structures" / "hits.jsonl").open() as handle:
        for line in handle:
            family = json.loads(line)["family"]
            counts[family] = counts.get(family, 0) + 1
    total = sum(counts.values())
    return {"total_hits": total, "counts": counts, "arithmetic_share": round(counts.get("arithmetic", 0) / total, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures_repo", type=Path, help="path to a Solomonoff-Figures checkout")
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    args = parser.parse_args()

    fig = args.figures_repo / "figures"
    result = {
        "source": {
            "paper": "arXiv:2609.30063v1",
            "figures_repo": "https://github.com/nourya-aliz/Solomonoff-Figures",
            "pinned_commit": PINNED_COMMIT,
            "checked_out_commit": check_commit(args.figures_repo),
            "compute_definition": "C = K * N * programs_per_round * 4096 * (round + 1), learner side only",
        },
        "shuffle_share_final_1M": shuffle_shares(fig),
        "shuffle_share_by_round_1M_K4": shuffle_share_by_round(fig),
        "matched_compute_best_bpb": matched_compute(fig),
        "frontier_end": frontier_end(fig),
        "dclm_24M_trajectory": text_trajectory(fig),
        "dclm_K4_plateau_by_rung": plateau_rounds(fig),
        "exponents": exponents(fig),
        "curriculum_value_fig3": curriculum_value(fig),
        "pre_pretraining_fig6": pre_pretraining(fig),
        "icl_best_accuracy": icl(fig),
        "discovered_structure": discovered_structure(fig),
    }
    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
