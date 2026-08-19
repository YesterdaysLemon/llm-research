from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
RESULT_PATH = EXPERIMENT / "results" / "smoke.json"
CONFIG_PATH = EXPERIMENT / "config" / "smoke.json"
FROZEN_COMMIT = "bc4ca00fdc4b3edbb06d387525220b940e1447c0"
FROZEN_CONFIG_SHA = "7bff46ca8ac07e92312ba167ada2012f21f20b0178ba8816793f16c76637d0c5"


class SmokeResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_provenance_is_clean_and_frozen(self) -> None:
        self.assertEqual(self.result["git_commit"], FROZEN_COMMIT)
        self.assertEqual(self.result["git_status_porcelain"], "")
        self.assertEqual(self.result["config_sha256"], FROZEN_CONFIG_SHA)
        actual = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual, FROZEN_CONFIG_SHA)

    def test_neither_capability_ladder_selected_a_rung(self) -> None:
        self.assertIsNone(self.result["teacher"]["training"]["selected_step"])
        self.assertIsNone(self.result["student"]["training"]["selected_step"])
        self.assertFalse(self.result["gates"]["all"])

    def test_natural_language_endpoint_learned(self) -> None:
        baseline = self.result["dataset"]["unigram_validation_nll"]
        for name in ("teacher", "student"):
            final = self.result[name]["training"]["rungs"][-1]
            self.assertLess(final["evaluation"]["natural"]["nll"], baseline - 0.2)

    def test_controlled_positive_controls_failed(self) -> None:
        teacher = self.result["teacher"]["training"]["rungs"][-1]
        student = self.result["student"]["training"]["rungs"][-1]
        self.assertLess(teacher["evaluation"]["controlled_id"]["accuracy"], 0.70)
        self.assertLess(student["evaluation"]["controlled_id"]["accuracy"], 0.55)


if __name__ == "__main__":
    unittest.main()

