from __future__ import annotations

import json
import unittest
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]


class E003ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(
            (EXPERIMENT / "results" / "confirm-label-geometry.json").read_text(
                encoding="utf-8"
            )
        )
        cls.analysis = json.loads(
            (EXPERIMENT / "results" / "analysis.json").read_text(encoding="utf-8")
        )

    def test_run_provenance_is_frozen_and_clean(self) -> None:
        self.assertEqual(self.result["git_status_porcelain"], "")
        self.assertEqual(
            self.result["config_sha256"],
            "a78736731718cbe4c610d75e0474261ea768d6df5c6310187307759761ed944c",
        )
        self.assertEqual(len(self.result["runs"]), 10)
        self.assertEqual(
            [row["seed"] for row in self.result["runs"]], list(range(5101, 5111))
        )

    def test_teacher_gate_passed(self) -> None:
        self.assertTrue(self.analysis["teacher_gate"]["passed"])

    def test_registered_primary_decision_is_positive(self) -> None:
        primary = self.analysis["primary_relational_minus_label_geometry"]["overall"]
        self.assertGreater(primary["ci95_low"], 0.0)
        self.assertEqual(
            self.analysis["registered_decision"],
            "learned_teacher_advantage_beyond_terminal_label_equivalence",
        )

    def test_label_geometry_underperformed_logits(self) -> None:
        secondary = self.analysis["secondary_label_geometry_minus_logits"]["overall"]
        self.assertLess(secondary["ci95_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
