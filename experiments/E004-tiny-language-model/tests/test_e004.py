from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data import (  # noqa: E402
    MODULUS,
    PairedBatchStream,
    Vocabulary,
    apply_operations,
    contains_forbidden_pair,
    controlled_record,
    encode_controlled_records,
    forbidden_occurrences,
    generate_controlled_records,
    generate_eval_records,
    tokenize,
)
from model import (  # noqa: E402
    TinyCausalLM,
    centered_relation,
    learning_rate_multiplier,
    relation_distance,
    target_relation,
)
from run import masked_cross_entropy  # noqa: E402


HELDOUT = (
    ("add:1", "multiply:2"),
    ("multiply:2", "add:1"),
    ("add:3", "multiply:3"),
    ("multiply:3", "add:3"),
)


def operations_from_prompt(prompt: str) -> list[str]:
    tokens = tokenize(prompt)
    return [
        f"{tokens[index + 1]}:{tokens[index + 2]}"
        for index, token in enumerate(tokens[:-2])
        if token == "applies"
    ]


class DataTests(unittest.TestCase):
    def test_affine_composition_is_order_sensitive(self) -> None:
        forward = apply_operations(3, ("add:1", "multiply:2"))
        reverse = apply_operations(3, ("multiply:2", "add:1"))
        self.assertEqual(forward, 8)
        self.assertEqual(reverse, 7)
        self.assertNotEqual(forward, reverse)
        self.assertEqual(apply_operations(0, ("subtract:2",)), MODULUS - 2)

    def test_training_records_exclude_heldout_bigrams(self) -> None:
        records = generate_controlled_records(
            count=100,
            depths=(2, 3, 4),
            seed=7,
            forbidden_pairs=HELDOUT,
        )
        for record in records:
            self.assertFalse(
                contains_forbidden_pair(
                    operations_from_prompt(record), set(HELDOUT)
                )
            )

    def test_heldout_eval_contains_exactly_one_registered_pair(self) -> None:
        rows = generate_eval_records(
            count=64,
            depths=(2, 3, 4, 6),
            seed=8,
            forbidden_pairs=HELDOUT,
            require_heldout=True,
        )
        self.assertEqual(len(rows), 64)
        for _, pair_label, prompt, _ in rows:
            occurrences = forbidden_occurrences(
                operations_from_prompt(prompt), set(HELDOUT)
            )
            self.assertEqual(len(occurrences), 1)
            self.assertEqual(pair_label, f"{occurrences[0][0]}->{occurrences[0][1]}")

    def test_heldout_depth_pair_grid_is_balanced(self) -> None:
        rows = generate_eval_records(
            count=64,
            depths=(2, 3, 4, 6),
            seed=8,
            forbidden_pairs=HELDOUT,
            require_heldout=True,
        )
        cells = Counter((depth, pair) for depth, pair, _, _ in rows)
        self.assertEqual(len(cells), 16)
        self.assertEqual(set(cells.values()), {4})

    def test_heldout_generation_preserves_configured_pair_order(self) -> None:
        rows = generate_eval_records(
            count=16,
            depths=(2, 3, 4, 6),
            seed=8,
            forbidden_pairs=HELDOUT,
            require_heldout=True,
        )
        observed = [rows[index * 4][1] for index in range(4)]
        expected = [f"{left}->{right}" for left, right in HELDOUT]
        self.assertEqual(observed, expected)

    def test_prompt_stops_immediately_before_answer(self) -> None:
        rows = generate_eval_records(
            count=1,
            depths=(2,),
            seed=9,
            forbidden_pairs=HELDOUT,
            require_heldout=True,
        )
        _, _, prompt, answer = rows[0]
        self.assertTrue(prompt.endswith("?"))
        self.assertIsInstance(answer, int)
        self.assertGreaterEqual(answer, 0)
        self.assertLess(answer, MODULUS)

    def test_paired_stream_aligns_controlled_records_and_masks_padding(self) -> None:
        vocabulary = Vocabulary(
            (
                "<pad>",
                "<unk>",
                "<bos>",
                "<eos>",
                "ava",
                "starts",
                "at",
                "applies",
                "add",
                "multiply",
                "subtract",
                "negate",
                "where",
                "is",
                ".",
                "?",
                *tuple(str(index) for index in range(MODULUS)),
            )
        )
        records = encode_controlled_records(
            [controlled_record("ava", 0, ("add:1",))], vocabulary
        )
        natural = torch.arange(200, dtype=torch.long) % len(vocabulary.tokens)
        stream = PairedBatchStream(
            natural,
            records,
            sequence_length=32,
            controlled_fraction=1.0,
            pad_id=vocabulary.stoi["<pad>"],
            question_id=vocabulary.stoi["?"],
            seed=12,
        )
        inputs, targets, mask, answer_mask, metadata = stream.batch(2)
        self.assertTrue(torch.all(inputs[:, 0].eq(vocabulary.stoi["<bos>"])))
        self.assertTrue(torch.all(targets[~mask].eq(vocabulary.stoi["<pad>"])))
        self.assertEqual(metadata["controlled_sequences"], 2)
        self.assertGreaterEqual(metadata["controlled_records"], 2)
        self.assertEqual(
            int(answer_mask.sum().item()), metadata["controlled_records"]
        )

    def test_sequence_mixture_is_exact_over_five_batches(self) -> None:
        vocabulary = Vocabulary(("<pad>", "<unk>", "<bos>", "<eos>", "x"))
        records = [torch.tensor([2, 4, 3], dtype=torch.long)]
        natural = torch.arange(500, dtype=torch.long) % 5
        stream = PairedBatchStream(
            natural,
            records,
            sequence_length=8,
            controlled_fraction=0.2,
            pad_id=0,
            question_id=4,
            seed=3,
        )
        total = 0
        for _ in range(5):
            _, _, _, _, metadata = stream.batch(16)
            total += metadata["controlled_sequences"]
        self.assertEqual(total, 16)


class ModelTests(unittest.TestCase):
    def test_answer_weight_changes_only_weighted_loss_reduction(self) -> None:
        logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
        targets = torch.tensor([[0, 0]])
        mask = torch.ones((1, 2), dtype=torch.bool)
        answer_mask = torch.tensor([[False, True]])
        ordinary = masked_cross_entropy(
            logits, targets, mask, answer_mask, answer_weight=1.0
        )
        weighted = masked_cross_entropy(
            logits, targets, mask, answer_mask, answer_weight=25.0
        )
        self.assertGreater(float(weighted), float(ordinary))

    def test_causal_lm_shapes(self) -> None:
        model = TinyCausalLM(
            vocab_size=37,
            sequence_length=16,
            d_model=24,
            n_heads=4,
            n_layers=3,
            d_ff=48,
            dropout=0.0,
        )
        logits, hidden = model(torch.randint(0, 37, (2, 11)), return_hidden=True)
        self.assertEqual(logits.shape, (2, 11, 37))
        self.assertEqual(len(hidden), 3)
        self.assertEqual(hidden[0].shape, (2, 11, 24))

    def test_causal_prefix_is_invariant_to_future_tokens(self) -> None:
        torch.manual_seed(4)
        model = TinyCausalLM(
            vocab_size=31,
            sequence_length=12,
            d_model=24,
            n_heads=4,
            n_layers=2,
            d_ff=48,
            dropout=0.0,
        ).eval()
        left = torch.tensor([[1, 2, 3, 4, 5, 6]])
        right = torch.tensor([[1, 2, 3, 9, 8, 7]])
        with torch.no_grad():
            left_logits, _ = model(left)
            right_logits, _ = model(right)
        torch.testing.assert_close(left_logits[:, :3], right_logits[:, :3])

    def test_relations_ignore_orthogonal_basis_change(self) -> None:
        torch.manual_seed(11)
        values = torch.randn(9, 7)
        q, _ = torch.linalg.qr(torch.randn(7, 7))
        torch.testing.assert_close(
            centered_relation(values),
            centered_relation(values @ q),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_target_relation_encodes_token_identity(self) -> None:
        labels = torch.tensor([2, 1, 2, 3])
        relation = target_relation(labels)
        self.assertLess(float(relation_distance(relation, relation)), 1e-8)
        self.assertGreater(float(relation[0, 2]), float(relation[0, 1]))

    def test_learning_rate_schedule_can_hold_constant_after_warmup(self) -> None:
        values = [
            learning_rate_multiplier(
                step, total_steps=100, warmup_steps=10, minimum_ratio=1.0
            )
            for step in range(100)
        ]
        self.assertLess(values[0], values[9])
        self.assertAlmostEqual(values[9], 1.0)
        self.assertAlmostEqual(values[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
