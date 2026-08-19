from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
ORDINARY = json.loads(
    (EXPERIMENT / "results" / "smoke.json").read_text(encoding="utf-8")
)
WEIGHTED_PATH = EXPERIMENT / "results" / "smoke-answer-weighted.json"
WEIGHTED = json.loads(WEIGHTED_PATH.read_text(encoding="utf-8"))
CONFIG_PATH = EXPERIMENT / "config" / "smoke-answer-weighted.json"
FROZEN_COMMIT = "74c18976a6692e9952372535739849da683844ff"
FROZEN_CONFIG_SHA = "893488f25cc7ae24ae2f64044944a998e6f234618afaf2713391bc0927a5aa1a"


class AnswerWeightedResultTests(unittest.TestCase):
    def test_provenance_is_clean_and_frozen(self) -> None:
        self.assertEqual(WEIGHTED["git_commit"], FROZEN_COMMIT)
        self.assertEqual(WEIGHTED["git_status_porcelain"], "")
        self.assertEqual(WEIGHTED["config_sha256"], FROZEN_CONFIG_SHA)
        self.assertEqual(
            hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(), FROZEN_CONFIG_SHA
        )

    def test_answer_exposure_matches_complete_records(self) -> None:
        for name in ("teacher", "student"):
            final = WEIGHTED[name]["training"]["rungs"][-1]["source_exposures"]
            self.assertEqual(
                final["controlled_answer_tokens"], final["controlled_records"]
            )

    def test_weighting_improved_shallow_id_accuracy(self) -> None:
        for name in ("teacher", "student"):
            ordinary = ORDINARY[name]["training"]["rungs"][-1]["evaluation"]
            weighted = WEIGHTED[name]["training"]["rungs"][-1]["evaluation"]
            gain = (
                weighted["controlled_id"]["accuracy_by_depth"]["2"]
                - ordinary["controlled_id"]["accuracy_by_depth"]["2"]
            )
            self.assertGreater(gain, 0.50)

    def test_positive_controls_still_failed(self) -> None:
        self.assertIsNone(WEIGHTED["teacher"]["training"]["selected_step"])
        self.assertIsNone(WEIGHTED["student"]["training"]["selected_step"])
        self.assertFalse(WEIGHTED["gates"]["all"])
        teacher = WEIGHTED["teacher"]["training"]["rungs"][-1]["evaluation"]
        self.assertLess(teacher["controlled_id"]["accuracy"], 0.70)
        self.assertLess(
            teacher["controlled_id"]["accuracy_by_depth"]["4"], 0.50
        )


if __name__ == "__main__":
    unittest.main()
