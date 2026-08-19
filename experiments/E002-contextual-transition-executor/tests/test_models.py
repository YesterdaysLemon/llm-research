from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "run_development.py"
SPEC = importlib.util.spec_from_file_location("run_e002", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_e002 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_e002
SPEC.loader.exec_module(run_e002)

STUDY_PATH = Path(__file__).resolve().parents[1] / "src" / "run_study.py"
STUDY_SPEC = importlib.util.spec_from_file_location("run_e002_study", STUDY_PATH)
assert STUDY_SPEC is not None and STUDY_SPEC.loader is not None
run_study = importlib.util.module_from_spec(STUDY_SPEC)
sys.modules[STUDY_SPEC.name] = run_study
STUDY_SPEC.loader.exec_module(run_study)


class ExecutorTests(unittest.TestCase):
    def test_transition_executor_shape_and_normalization(self) -> None:
        model = run_e002.TransitionTableExecutor(7, 3, 11)
        tokens = torch.tensor([[10, 2, 7, 8, 11], [10, 4, 9, 11, 11]])
        logits, trajectory = model(tokens)
        self.assertEqual(logits.shape, (2, 7))
        self.assertEqual(trajectory.shape, (2, 3, 7))
        torch.testing.assert_close(logits.exp().sum(dim=-1), torch.ones(2))

    def test_transition_executor_uses_one_matrix_per_step(self) -> None:
        model = run_e002.TransitionTableExecutor(7, 3, 11)
        self.assertEqual(run_e002.p0.count_parameters(model), 147)
        self.assertEqual(model.active_parameters_per_step, 49)

    def test_transition_executor_can_encode_exact_permutations(self) -> None:
        table, _ = run_e002.p0.make_relation_table(7, 3, 13)
        model = run_e002.TransitionTableExecutor(7, 3, 11)
        with torch.no_grad():
            model.transition_logits.fill_(-20.0)
            for relation in range(3):
                for source in range(7):
                    model.transition_logits[relation, source, table[relation, source]] = 20.0
        tokens = torch.tensor([[10, 2, 7, 8, 9]])
        logits, _ = model(tokens)
        expected = run_e002.p0.apply_relations(2, [0, 1, 2], table)
        self.assertEqual(int(logits.argmax(dim=-1).item()), expected)

    def test_gru_executor_reuses_cell_parameters(self) -> None:
        model = run_e002.GRUExecutor(7, 3, 11, 8)
        expected = sum(parameter.numel() for parameter in model.cell.parameters())
        self.assertEqual(model.active_parameters_per_step, expected)
        tokens = torch.tensor([[10, 2, 7, 8, 11]])
        logits, trajectory = model(tokens)
        self.assertEqual(logits.shape, (1, 7))
        self.assertEqual(trajectory.shape, (1, 3, 8))

    def test_bitwise_study_relations_are_permutations(self) -> None:
        config = {
            "task_family": "bitwise",
            "data_seed": 27182,
            "data": {"num_entities": 64, "num_relations": 8},
        }
        table, _ = run_study.make_task_table(config)
        expected = list(range(64))
        for row in table:
            self.assertEqual(sorted(row.tolist()), expected)

    def test_label_geometry_repeats_terminal_class_across_layers(self) -> None:
        targets = run_study.label_geometry_targets(
            torch.tensor([2, 0, 2]), num_classes=4, steps=2
        )
        self.assertEqual(targets.shape, (3, 2, 4))
        torch.testing.assert_close(targets[:, 0], targets[:, 1])
        self.assertEqual(targets[0, 0].tolist(), [0.0, 0.0, 1.0, 0.0])
        torch.testing.assert_close(targets[0], targets[2])
        self.assertFalse(torch.equal(targets[0], targets[1]))


if __name__ == "__main__":
    unittest.main()
