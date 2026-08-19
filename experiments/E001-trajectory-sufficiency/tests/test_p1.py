from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "run_p1.py"
SPEC = importlib.util.spec_from_file_location("run_p1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_p1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_p1
SPEC.loader.exec_module(run_p1)


class P1Tests(unittest.TestCase):
    def test_bitwise_relations_are_permutations(self) -> None:
        config = {
            "task_family": "bitwise",
            "data_seed": 31415,
            "data": {"num_entities": 64, "num_relations": 8},
        }
        table, definitions = run_p1.make_task_table(config)
        expected = list(range(64))
        self.assertEqual(len(definitions), 8)
        for row in table:
            self.assertEqual(sorted(row.tolist()), expected)

    def test_affine_family_matches_pilot_generator(self) -> None:
        config = {
            "task_family": "affine",
            "data_seed": 2718,
            "data": {"num_entities": 47, "num_relations": 8},
        }
        actual, _ = run_p1.make_task_table(config)
        expected, _ = run_p1.p0.make_relation_table(47, 8, 2718)
        np.testing.assert_array_equal(actual, expected)

    def test_self_relation_penalizes_layer_change(self) -> None:
        generator = torch.Generator().manual_seed(9)
        first = torch.randn(32, 16, generator=generator)
        second = torch.randn(32, 16, generator=generator)
        trajectory = torch.stack((first, second), dim=1).requires_grad_()
        loss = run_p1.self_relational_loss(trajectory)
        self.assertGreater(float(loss.detach()), 0.01)
        loss.backward()
        self.assertIsNotNone(trajectory.grad)

    def test_dataset_byte_count_is_exact(self) -> None:
        table, _ = run_p1.p0.make_relation_table(23, 4, 17)
        dataset = run_p1.p0.CompositionDataset(
            count=100,
            depths=[2, 3, 4],
            table=table,
            seed=18,
            max_depth=4,
        )
        expected = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (dataset.tokens, dataset.labels, dataset.depths)
        )
        self.assertEqual(run_p1.dataset_bytes(dataset), expected)


if __name__ == "__main__":
    unittest.main()
