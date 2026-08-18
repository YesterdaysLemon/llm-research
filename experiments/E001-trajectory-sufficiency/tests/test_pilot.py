from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "run_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_pilot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_pilot
SPEC.loader.exec_module(run_pilot)


class PilotTests(unittest.TestCase):
    def test_relation_table_contains_permutations(self) -> None:
        table, _ = run_pilot.make_relation_table(23, 4, 1729)
        expected = list(range(23))
        for row in table:
            self.assertEqual(sorted(row.tolist()), expected)

    def test_apply_relations_matches_manual_lookup(self) -> None:
        table, _ = run_pilot.make_relation_table(23, 4, 1729)
        expected = int(table[3, int(table[1, int(table[0, 7])])])
        self.assertEqual(run_pilot.apply_relations(7, [0, 1, 3], table), expected)

    def test_dataset_labels_match_tokens(self) -> None:
        table, _ = run_pilot.make_relation_table(23, 4, 1729)
        dataset = run_pilot.CompositionDataset(
            count=50,
            depths=[1, 2, 3, 4],
            table=table,
            seed=1730,
            max_depth=6,
        )
        for _, tokens, label, depth in dataset:
            start = int(tokens[1])
            relation_tokens = tokens[2 : 2 + int(depth)]
            relations = (relation_tokens - dataset.num_entities).tolist()
            self.assertEqual(
                run_pilot.apply_relations(start, relations, table), int(label)
            )

    def test_pair_constraints_are_enforced(self) -> None:
        table, _ = run_pilot.make_relation_table(23, 4, 1729)
        forbidden = run_pilot.CompositionDataset(
            count=100,
            depths=[2, 3, 4],
            table=table,
            seed=1730,
            max_depth=4,
            forbidden_pairs=[[0, 1]],
        )
        required = run_pilot.CompositionDataset(
            count=100,
            depths=[2, 3, 4],
            table=table,
            seed=1731,
            max_depth=4,
            required_pairs=[[0, 1]],
        )

        def contains_pair(dataset: object, tokens: torch.Tensor, depth: int) -> bool:
            relations = (
                tokens[2 : 2 + depth] - dataset.num_entities  # type: ignore[attr-defined]
            ).tolist()
            return any(
                left == 0 and right == 1
                for left, right in zip(relations[:-1], relations[1:], strict=True)
            )

        self.assertTrue(
            all(
                not contains_pair(forbidden, tokens, int(depth))
                for _, tokens, _, depth in forbidden
            )
        )
        self.assertTrue(
            all(
                contains_pair(required, tokens, int(depth))
                for _, tokens, _, depth in required
            )
        )

    def test_gram_is_orthogonally_invariant(self) -> None:
        generator = torch.Generator().manual_seed(4)
        features = torch.randn(32, 12, generator=generator)
        rotation_source = torch.randn(12, 12, generator=generator)
        rotation, _ = torch.linalg.qr(rotation_source)
        original = run_pilot.normalized_gram(features)
        rotated = run_pilot.normalized_gram(features @ rotation)
        torch.testing.assert_close(original, rotated, atol=1e-5, rtol=1e-5)

    def test_shuffling_changes_relational_target(self) -> None:
        generator = torch.Generator().manual_seed(8)
        features = torch.randn(32, 12, generator=generator)
        permutation = torch.randperm(32, generator=generator)
        original = run_pilot.normalized_gram(features)
        shuffled = run_pilot.normalized_gram(features[permutation])
        self.assertGreater(float((original - shuffled).pow(2).sum()), 0.1)


if __name__ == "__main__":
    unittest.main()
