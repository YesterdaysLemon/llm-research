from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HEART_FAMILIES = {
    "red": ["❤️", "❤"],
    "orange": ["🧡"],
    "yellow": ["💛"],
    "green": ["💚"],
    "blue": ["💙"],
    "light_blue": ["🩵"],
    "purple": ["💜"],
    "pink": ["🩷"],
    "black": ["🖤"],
    "grey": ["🩶"],
    "white": ["🤍"],
    "brown": ["🤎"],
    "suit": ["♥️", "♥"],
    "broken": ["💔"],
    "fire": ["❤️\u200d🔥", "❤\u200d🔥"],
    "mending": ["❤️\u200d🩹", "❤\u200d🩹"],
    "sparkling": ["💖"],
    "growing": ["💗"],
    "beating": ["💓"],
    "revolving": ["💞"],
    "two_hearts": ["💕"],
    "arrow": ["💘"],
    "ribbon": ["💝"],
    "decoration": ["💟"],
    "exclamation": ["❣️", "❣"],
    "anatomical": ["🫀"],
}
GLYPH_TO_FAMILY = {
    glyph: family for family, glyphs in HEART_FAMILIES.items() for glyph in glyphs
}
HEART_GLYPHS = sorted(GLYPH_TO_FAMILY, key=len, reverse=True)
NARROW_FAMILIES = {
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "light_blue",
    "purple",
    "pink",
    "black",
    "grey",
    "white",
    "brown",
}


def tokenize_line(line: str) -> tuple[list[str], str, str]:
    hearts: list[str] = []
    invalid: list[str] = []
    index = 0
    while index < len(line):
        matched = next((glyph for glyph in HEART_GLYPHS if line.startswith(glyph, index)), None)
        if matched:
            hearts.append(matched)
            index += len(matched)
        else:
            invalid.append(line[index])
            index += 1
    leading = line[: len(line) - len(line.lstrip(" \t"))]
    return hearts, "".join(invalid), leading


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def bootstrap_mean_ci95(values: list[float], label: str, draws: int = 10_000) -> list[float] | None:
    if not values:
        return None
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    sample_size = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(sample_size)) / sample_size
        for _ in range(draws)
    )
    return [means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]]


def best_periodic_fit(matrix: list[list[str]]) -> dict[str, Any] | None:
    if not matrix or len({cell for row in matrix for cell in row}) < 2:
        return None
    rows = len(matrix)
    cols = len(matrix[0])
    best: dict[str, Any] | None = None
    for slope in (-1, 0, 1):
        for period in range(2, 7):
            buckets: dict[int, list[str]] = defaultdict(list)
            for row in range(rows):
                for col in range(cols):
                    buckets[(col - slope * row) % period].append(matrix[row][col])
            template = {bucket: Counter(values).most_common(1)[0][0] for bucket, values in buckets.items()}
            if len(set(template.values())) < 2:
                continue
            matches = sum(
                matrix[row][col] == template[(col - slope * row) % period]
                for row in range(rows)
                for col in range(cols)
            )
            fit = matches / (rows * cols)
            candidate = {"fit": fit, "period": period, "slope": slope}
            if best is None or (fit, -period, -abs(slope)) > (
                best["fit"],
                -best["period"],
                -abs(best["slope"]),
            ):
                best = candidate
    return best


def analyze_text(text: str | None) -> dict[str, Any]:
    if text is None:
        return {"has_response": False}
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    lines = normalized.split("\n") if normalized else []
    opening_line = lines[0] if lines else ""
    opening_hearts, opening_invalid, opening_leading = tokenize_line(opening_line)
    structural_wall_lines = [line for line in lines[1:] if line != ""]
    parsed_rows = [tokenize_line(line) for line in structural_wall_lines]
    wall_rows = [row[0] for row in parsed_rows]
    wall_invalid = [row[1] for row in parsed_rows]
    wall_leading = [row[2] for row in parsed_rows]
    row_lengths = [len(row) for row in wall_rows]
    wall_glyphs = [glyph for row in wall_rows for glyph in row]
    wall_families = [GLYPH_TO_FAMILY[glyph] for glyph in wall_glyphs]
    wall_parseable = bool(wall_rows) and all(not value for value in wall_invalid) and all(
        not value for value in wall_leading
    )
    rectangular = wall_parseable and bool(row_lengths) and len(set(row_lengths)) == 1 and row_lengths[0] > 0
    matrix = wall_rows if rectangular else []

    adjacent_h: list[bool] = []
    adjacent_v: list[bool] = []
    if matrix:
        adjacent_h = [
            row[col] != row[col + 1] for row in matrix for col in range(len(row) - 1)
        ]
        adjacent_v = [
            matrix[row][col] != matrix[row + 1][col]
            for row in range(len(matrix) - 1)
            for col in range(len(matrix[0]))
        ]

    checkerboard_exact = False
    if matrix and len(set(wall_glyphs)) == 2:
        parity_zero = {matrix[r][c] for r in range(len(matrix)) for c in range(len(matrix[0])) if (r + c) % 2 == 0}
        parity_one = {matrix[r][c] for r in range(len(matrix)) for c in range(len(matrix[0])) if (r + c) % 2 == 1}
        checkerboard_exact = len(parity_zero) == 1 and len(parity_one) == 1 and parity_zero != parity_one

    periodic = best_periodic_fit(matrix)
    dominant_glyph = Counter(wall_glyphs).most_common(1)[0][0] if wall_glyphs else None
    dominant_family = Counter(wall_families).most_common(1)[0][0] if wall_families else None
    opening_glyph = opening_hearts[0] if len(opening_hearts) == 1 else None
    opening_family = GLYPH_TO_FAMILY.get(opening_glyph) if opening_glyph else None
    no_blank_separator = len(lines) >= 2 and lines[1] != ""
    no_extra_text = not opening_invalid and all(not value for value in wall_invalid)
    only_linebreak_separators = no_extra_text and not opening_leading and all(not value for value in wall_leading)

    return {
        "has_response": True,
        "line_count": len(lines),
        "opening_glyph": opening_glyph,
        "opening_family": opening_family,
        "opening_exactly_one_heart": len(opening_hearts) == 1 and not opening_invalid and not opening_leading,
        "wall_begins_next_line": no_blank_separator,
        "wall_rows": len(wall_rows),
        "wall_total_hearts": len(wall_glyphs),
        "wall_min_columns": min(row_lengths) if row_lengths else 0,
        "wall_median_columns": statistics.median(row_lengths) if row_lengths else 0,
        "wall_max_columns": max(row_lengths) if row_lengths else 0,
        "wall_rectangular": rectangular,
        "wall_occupancy": (len(wall_glyphs) / (len(wall_rows) * max(row_lengths))) if row_lengths and max(row_lengths) else 0.0,
        "wall_parseable": wall_parseable,
        "wall_only_hearts_and_linebreaks": only_linebreak_separators and bool(wall_glyphs),
        "no_prose_or_fences": no_extra_text and "```" not in normalized,
        "wall_distinct_glyphs": len(set(wall_glyphs)),
        "wall_distinct_families": len(set(wall_families)),
        "wall_dominant_glyph": dominant_glyph,
        "wall_dominant_family": dominant_family,
        "wall_matches_opening_fraction": (
            sum(glyph == opening_glyph for glyph in wall_glyphs) / len(wall_glyphs)
            if wall_glyphs and opening_glyph
            else None
        ),
        "wall_glyph_entropy_bits": entropy(wall_glyphs),
        "wall_family_entropy_bits": entropy(wall_families),
        "horizontal_disagreement": statistics.mean(adjacent_h) if adjacent_h else None,
        "vertical_disagreement": statistics.mean(adjacent_v) if adjacent_v else None,
        "checkerboard_exact": checkerboard_exact,
        "periodic_fit": periodic["fit"] if periodic else None,
        "periodic_period": periodic["period"] if periodic else None,
        "periodic_slope": periodic["slope"] if periodic else None,
        "barber_pole_exact": bool(periodic and periodic["fit"] == 1.0 and periodic["slope"] != 0),
        "leading_indent_rows": sum(bool(value) for value in wall_leading),
        "narrow_conventional_only": bool(wall_families) and all(family in NARROW_FAMILIES for family in wall_families),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    enriched: list[dict[str, Any]] = []
    for record in records:
        metrics = analyze_text(record.get("final_text"))
        combined = {**record, **metrics}
        enriched.append(combined)
        groups[(record["provider"], record["model"], record["effort"])].append(combined)

    summary_groups: list[dict[str, Any]] = []
    for (provider, model, effort), rows in sorted(groups.items()):
        successful = [row for row in rows if row.get("exit_code") == 0 and row.get("has_response")]
        parseable = [row for row in successful if row.get("wall_parseable")]
        opening_counts = Counter(row.get("opening_family") or "unparsed" for row in successful)
        dominant_counts = Counter(row.get("wall_dominant_family") or "unparsed" for row in successful)
        dimension_counts = Counter(
            f"{row['wall_rows']}×{row['wall_max_columns']}" for row in parseable
        )
        strict_count = sum(
            bool(row.get("opening_exactly_one_heart"))
            and bool(row.get("wall_begins_next_line"))
            and bool(row.get("wall_only_hearts_and_linebreaks"))
            and bool(row.get("no_prose_or_fences"))
            for row in successful
        )
        reasoning_tokens = []
        for row in successful:
            usage = row.get("usage") or {}
            details = usage.get("output_tokens_details") or {}
            reasoning_tokens.append(
                usage.get("reasoning_output_tokens", details.get("thinking_tokens", 0)) or 0
            )
        wall_heart_counts = [float(row["wall_total_hearts"]) for row in parseable]
        summary_groups.append(
            {
                "provider": provider,
                "model": model,
                "effort": effort,
                "attempted": len(rows),
                "successful": len(successful),
                "parseable_walls": len(parseable),
                "opening_family_counts": dict(opening_counts),
                "dominant_wall_family_counts": dict(dominant_counts),
                "dimension_counts": dict(sorted(dimension_counts.items())),
                "unique_response_count": len({row.get("final_text") for row in successful}),
                "mean_wall_rows": statistics.mean(row["wall_rows"] for row in parseable) if parseable else None,
                "mean_wall_hearts": statistics.mean(wall_heart_counts) if wall_heart_counts else None,
                "mean_wall_hearts_ci95": bootstrap_mean_ci95(
                    wall_heart_counts, f"{provider}|{model}|{effort}|wall_hearts"
                ),
                "median_duration_seconds": statistics.median(row["duration_seconds"] for row in successful) if successful else None,
                "mean_reasoning_tokens": statistics.mean(reasoning_tokens) if reasoning_tokens else None,
                "rectangular_rate_all_successes": sum(bool(row.get("wall_rectangular")) for row in successful) / len(successful) if successful else None,
                "multicolor_rate_all_successes": sum((row.get("wall_distinct_families") or 0) > 1 for row in successful) / len(successful) if successful else None,
                "checkerboard_rate_all_successes": sum(bool(row.get("checkerboard_exact")) for row in successful) / len(successful) if successful else None,
                "barber_pole_rate_all_successes": sum(bool(row.get("barber_pole_exact")) for row in successful) / len(successful) if successful else None,
                "blank_separator_rate": sum(not bool(row.get("wall_begins_next_line")) for row in successful) / len(successful) if successful else None,
                "strict_compliance_count": strict_count,
                "strict_compliance_rate": strict_count / len(successful) if successful else None,
            }
        )
    return {"groups": summary_groups, "runs": enriched}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for input_path in args.input
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    analysis = summarize(records)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "schema": 1,
        "input_records": len(records),
        "groups": analysis["groups"],
    }
    args.json.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    excluded_csv_fields = {
        "stdout",
        "stderr",
        "final_text",
        "usage",
        "model_usage",
        "parse_errors",
    }
    fields = sorted(
        {key for row in analysis["runs"] for key in row if key not in excluded_csv_fields}
    )
    with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(analysis["runs"])


if __name__ == "__main__":
    main()
