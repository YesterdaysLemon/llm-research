"""Fail-closed checks for generated analysis and the release PDF."""

from __future__ import annotations

import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output" / "pdf" / "relational_activation_distillation_preprint.pdf"
STATS = ROOT / "paper" / "generated" / "statistics.json"
MANUSCRIPT = ROOT / "paper" / "preprint.md"


def close(actual: float, expected: float, tolerance: float = 0.00005) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    payload = json.loads(STATS.read_text(encoding="utf-8"))
    close(payload["affine"]["conditions"]["transformer-relational"]["accuracy"]["mean"], 0.23309)
    close(payload["affine"]["conditions"]["transformer-logits"]["accuracy"]["mean"], 0.11298)
    close(payload["affine"]["conditions"]["table-labels"]["accuracy"]["mean"], 1.0)
    close(payload["affine"]["paired"]["relational_minus_logits_overall"]["mean"], 0.12011)
    close(payload["teacher_gates"]["bitwise"]["depth4"], 0.8921)
    close(
        payload["affine"]["conditions"]["transformer-relational"]["id_depth"]["4"]["mean"],
        0.0815,
    )
    close(
        payload["affine"]["conditions"]["transformer-relational"]["normalized_transfer"]["4"],
        0.230,
        tolerance=0.001,
    )
    close(
        payload["e003_label_geometry"]["primary_relational_minus_label_geometry"]["overall"]["mean"],
        0.17439,
    )
    close(
        payload["e003_label_geometry"]["secondary_label_geometry_minus_logits"]["overall"]["mean"],
        -0.05428,
    )
    assert payload["teacher_gates"]["bitwise"]["passed_registered_95_percent_floor"] is False

    source = MANUSCRIPT.read_text(encoding="utf-8")
    for required in (
        "23.31%",
        "12.01 percentage points",
        "61.3 times fewer",
        "The entire bitwise distillation cell is formally invalid",
        "target-label Gram",
        "severe student capability ceiling",
        "17.44 percentage points",
    ):
        assert required in source, required
    for forbidden in ("TODO", "TBD", "PLACEHOLDER"):
        assert forbidden not in source

    reader = PdfReader(str(PDF))
    assert len(reader.pages) == 12
    assert reader.metadata.title == (
        "Relational Layer Geometry Improves Compositional Transfer "
        "Under a Capability Ceiling"
    )
    extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
    assert len(extracted) > 35_000
    for required in (
        "23.31%",
        "17.44",
        "61.3 times fewer",
        "teacher gate failed",
    ):
        assert required in extracted, required
    assert sum(len(page.get("/Annots", [])) for page in reader.pages) >= 10

    with pdfplumber.open(str(PDF)) as document:
        assert len(document.pages) == 12
        assert all(len(page.extract_text() or "") > 1_000 for page in document.pages)

    print("paper validation passed")


if __name__ == "__main__":
    main()
