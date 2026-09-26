from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from candidates import CandidatePool, unigram_nll, validation_windows  # noqa: E402
from model import TinyCausalLM  # noqa: E402  (E004)
from selection import (  # noqa: E402
    ARMS,
    Direction,
    SnapshotBank,
    adam_step_operator,
    arm_scores,
    cosine_from,
    jvp_scores,
    learning_direction,
    parameter_dict,
    per_sample_statistics,
    sequence_losses,
    spearman,
    top_indices,
)
from train import (  # noqa: E402
    flops_matched_steps,
    forward_flops_per_token,
    run_one,
    schedule,
    step_flops,
)

CONFIG_DIR = EXPERIMENT / "config"


def tiny_model(seed: int = 0) -> TinyCausalLM:
    torch.manual_seed(seed)
    return TinyCausalLM(
        vocab_size=40,
        sequence_length=8,
        d_model=16,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        dropout=0.0,
    )


def random_tangent(params: dict[str, torch.Tensor], scale: float = 1e-3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    return {
        name: torch.randn(value.shape, generator=generator, dtype=value.dtype) * scale
        for name, value in params.items()
    }


class ScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = tiny_model()
        self.params = parameter_dict(self.model)
        generator = torch.Generator().manual_seed(3)
        self.inputs = torch.randint(0, 40, (6, 8), generator=generator)
        self.targets = torch.randint(0, 40, (6, 8), generator=generator)
        self.tangent = random_tangent(self.params)
        self.operator = {name: torch.ones_like(value) for name, value in self.params.items()}

    def explicit_gradients(self) -> list[dict[str, torch.Tensor]]:
        rows = []
        for index in range(self.inputs.shape[0]):
            self.model.zero_grad(set_to_none=True)
            loss = sequence_losses(
                self.model, self.params, self.inputs[index : index + 1], self.targets[index : index + 1]
            )[0]
            loss.backward()
            rows.append({name: value.grad.detach().clone() for name, value in self.params.items()})
        self.model.zero_grad(set_to_none=True)
        return rows

    def test_tied_embedding_appears_once(self) -> None:
        self.assertIn("token_embedding.weight", self.params)
        self.assertNotIn("lm_head.weight", self.params)

    def test_jvp_matches_explicit_gradient_dot_products(self) -> None:
        with sdpa_kernel(SDPBackend.MATH):
            _, scores = jvp_scores(self.model, self.params, self.tangent, self.inputs, self.targets)
            gradients = self.explicit_gradients()
        expected = torch.stack(
            [sum((row[name] * self.tangent[name]).sum() for name in row) for row in gradients]
        )
        torch.testing.assert_close(scores, expected, rtol=1e-4, atol=1e-8)

    def test_jvp_matches_central_difference(self) -> None:
        model = tiny_model().double()
        params = parameter_dict(model)
        tangent = random_tangent(params, scale=1.0)
        epsilon = 1e-6
        with sdpa_kernel(SDPBackend.MATH):
            _, scores = jvp_scores(model, params, tangent, self.inputs, self.targets)
            plus = {name: value.detach() + epsilon * tangent[name] for name, value in params.items()}
            minus = {name: value.detach() - epsilon * tangent[name] for name, value in params.items()}
            difference = (
                sequence_losses(model, plus, self.inputs, self.targets)
                - sequence_losses(model, minus, self.inputs, self.targets)
            ) / (2 * epsilon)
        torch.testing.assert_close(scores, difference.detach(), rtol=1e-5, atol=1e-8)

    def test_per_sample_statistics_match_explicit_gradients(self) -> None:
        operator = {name: torch.rand_like(value) + 0.5 for name, value in self.params.items()}
        direction = Direction(self.tangent, operator, 0, 1.0)
        with sdpa_kernel(SDPBackend.MATH):
            statistics = per_sample_statistics(
                self.model, self.params, direction, self.inputs, self.targets, chunk=4
            )
            gradients = self.explicit_gradients()
        norms = torch.stack(
            [sum((operator[name] * row[name].square()).sum() for name in row).sqrt() for row in gradients]
        )
        dots = torch.stack(
            [sum((row[name] * self.tangent[name]).sum() for name in row) for row in gradients]
        )
        torch.testing.assert_close(statistics["gradnorm"], norms, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(statistics["dot"], dots, rtol=1e-4, atol=1e-8)

    def test_adam_step_operator_matches_bias_corrected_second_moment(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.1)
        for _ in range(3):
            optimizer.zero_grad()
            sequence_losses(self.model, self.params, self.inputs, self.targets).mean().backward()
            optimizer.step()
        operator = adam_step_operator(optimizer, self.params)
        for name, parameter in self.params.items():
            state = optimizer.state[parameter]
            v_hat = state["exp_avg_sq"] / (1 - 0.999**3)
            torch.testing.assert_close(operator[name], 0.01 / (v_hat.sqrt() + 1e-8))

    def test_direction_norm_and_cosine_are_consistent(self) -> None:
        anchor = {name: value.detach() + 0.01 for name, value in self.params.items()}
        operator = {name: torch.full_like(value, 2.0) for name, value in self.params.items()}
        direction = learning_direction(self.params, anchor, operator, anchor_step=4)
        count = sum(value.numel() for value in self.params.values())
        self.assertAlmostEqual(direction.preconditioned_norm, (2.0 * 1e-4 * count) ** 0.5, places=6)
        self.assertEqual(direction.anchor_step, 4)
        # |cos| of a vector with itself is one.
        cosine = cosine_from(torch.tensor([3.0]), torch.tensor([3.0]), 1.0)
        self.assertAlmostEqual(float(cosine), 1.0)


class SelectionTests(unittest.TestCase):
    def test_top_indices_break_ties_by_pool_order(self) -> None:
        scores = torch.tensor([1.0, 5.0, 5.0, 0.0, 5.0])
        self.assertEqual(top_indices(scores, 2).tolist(), [1, 2])

    def test_arm_scores(self) -> None:
        statistics = {
            "loss": torch.tensor([1.0, 2.0]),
            "gradnorm": torch.tensor([2.0, 1.0]),
            "dot": torch.tensor([-1.0, 0.5]),
            "jvp": torch.tensor([-3.0, 1.0]),
        }
        self.assertEqual(arm_scores("loss", statistics, 1.0).tolist(), [1.0, 2.0])
        self.assertEqual(arm_scores("signed", statistics, 1.0).tolist(), [-3.0, 1.0])
        self.assertEqual(arm_scores("absolute", statistics, 1.0).tolist(), [3.0, 1.0])
        self.assertEqual(arm_scores("cosine", statistics, 1.0).tolist(), [0.5, 0.5])
        with self.assertRaises(ValueError):
            arm_scores("uniform", statistics, 1.0)

    def test_spearman(self) -> None:
        values = torch.tensor([0.1, 0.4, 0.2, 0.9])
        self.assertAlmostEqual(spearman(values, values * 3 + 1), 1.0)
        self.assertAlmostEqual(spearman(values, -values), -1.0)


class SnapshotTests(unittest.TestCase):
    def test_anchor_is_latest_snapshot_at_or_before_half(self) -> None:
        bank = SnapshotBank(interval=10)
        self.assertEqual(bank.anchor_step(1), 0)
        self.assertEqual(bank.anchor_step(39), 10)
        self.assertEqual(bank.anchor_step(40), 20)
        self.assertEqual(bank.anchor_step(41), 20)

    def test_store_and_prune_keep_every_future_anchor(self) -> None:
        bank = SnapshotBank(interval=5)
        value = {"w": torch.zeros(2)}
        for step in range(0, 101):
            bank.store(step, {"w": value["w"] + step})
            bank.prune(step)
            anchor_step, anchor = bank.anchor(step, torch.device("cpu"))
            self.assertEqual(anchor_step, bank.anchor_step(step))
            self.assertEqual(float(anchor["w"][0]), float(anchor_step))
        self.assertEqual(min(bank.snapshots), bank.anchor_step(100))


class CandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = torch.arange(4, 504) % 60 + 4

    def test_pool_is_deterministic(self) -> None:
        left = CandidatePool(self.stream, sequence_length=8, vocab_size=64, seed=1)
        right = CandidatePool(self.stream, sequence_length=8, vocab_size=64, seed=1)
        for _ in range(3):
            for a, b in zip(left.draw(16), right.draw(16)):
                self.assertTrue(torch.equal(a, b))

    def test_targets_are_next_tokens(self) -> None:
        pool = CandidatePool(self.stream, sequence_length=8, vocab_size=64, seed=1)
        inputs, targets, noise = pool.draw(4)
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))
        self.assertFalse(bool(noise.any()))

    def test_noise_replaces_windows_without_moving_clean_ones(self) -> None:
        clean = CandidatePool(self.stream, sequence_length=8, vocab_size=64, seed=1, noise_seed=9)
        noisy = CandidatePool(
            self.stream, sequence_length=8, vocab_size=64, seed=1, noise_fraction=0.25, noise_seed=9
        )
        replaced = 0
        total = 0
        for _ in range(50):
            clean_inputs, _, _ = clean.draw(64)
            noisy_inputs, _, noise = noisy.draw(64)
            self.assertTrue(torch.equal(clean_inputs[~noise], noisy_inputs[~noise]))
            self.assertTrue(bool((noisy_inputs[noise] >= 4).all()))
            replaced += int(noise.sum())
            total += noise.numel()
        self.assertAlmostEqual(replaced / total, 0.25, delta=0.03)

    def test_validation_windows_are_non_overlapping(self) -> None:
        windows = validation_windows(torch.arange(20), 4)
        self.assertEqual(windows.shape, (4, 5))
        self.assertEqual(windows[1].tolist(), [4, 5, 6, 7, 8])
        self.assertGreater(unigram_nll(self.stream, windows % 60 + 4, 64), 0.0)


class ConfigTests(unittest.TestCase):
    def load(self, name: str) -> dict:
        return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))

    def test_fixed_config_is_the_registered_design(self) -> None:
        config = self.load("fixed.json")
        self.assertEqual(config["arms"], list(ARMS))
        self.assertEqual(len(config["seeds"]), 5)
        self.assertEqual([c["id"] for c in config["conditions"]], ["clean", "noisy"])
        self.assertEqual(config["selection_steps"], 6000)
        self.assertEqual(config["uniform_steps"], 22000)
        runs = schedule(config)
        self.assertEqual(len(runs), 60)
        self.assertEqual(
            flops_matched_steps(6000, batch=16, pool=64), config["uniform_steps"]
        )

    def test_smoke_config_is_valid(self) -> None:
        config = self.load("smoke.json")
        runs = schedule(config)
        self.assertEqual(len(runs), len(ARMS) * len(config["conditions"]) * len(config["seeds"]))

    def test_configs_share_everything_but_budgets_and_seeds(self) -> None:
        fixed, smoke = self.load("fixed.json"), self.load("smoke.json")
        for key in ("dataset", "student", "training", "selection", "arms", "conditions", "gates"):
            self.assertEqual(fixed[key], smoke[key], key)

    def test_step_flops_ratio(self) -> None:
        spec = {"d_model": 160, "n_layers": 4, "d_ff": 640}
        forward = forward_flops_per_token(spec, vocab_size=2048, sequence_length=64)
        uniform = step_flops("uniform", forward_per_token=forward, batch=16, pool=64, length=64, scored=False)
        absolute = step_flops("absolute", forward_per_token=forward, batch=16, pool=64, length=64, scored=True)
        self.assertAlmostEqual(absolute / uniform, 11 / 3)


class EndToEndTests(unittest.TestCase):
    """Every arm runs on a synthetic corpus; this checks code paths, not effects."""

    def config(self) -> dict:
        config = json.loads((CONFIG_DIR / "smoke.json").read_text(encoding="utf-8"))
        config["student"] = {**config["student"], "d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32}
        config["dataset"] = {**config["dataset"], "sequence_length": 8}
        config["training"] = {**config["training"], "batch_size": 4, "use_bfloat16": False}
        config["selection"] = {
            **config["selection"],
            "snapshot_interval": 2,
            "diagnostic_interval": 3,
            "per_sample_chunk": 8,
        }
        config["evaluation"] = {**config["evaluation"], "batch_size": 8}
        return config

    def data(self) -> dict:
        class Vocabulary:
            tokens = tuple(f"t{index}" for index in range(30))

        generator = torch.Generator().manual_seed(5)
        train = torch.randint(4, 30, (600,), generator=generator)
        validation = torch.randint(4, 30, (161,), generator=generator)
        return {
            "vocabulary": Vocabulary(),
            "train": train,
            "validation_windows": validation_windows(validation, 8),
        }

    def test_every_arm_runs_and_is_deterministic(self) -> None:
        config, data = self.config(), self.data()
        condition = {"id": "noisy", "noise_fraction": 0.25}
        results = {}
        with sdpa_kernel(SDPBackend.MATH):
            for arm in ARMS:
                results[arm] = run_one(
                    condition=condition,
                    arm=arm,
                    seed=1,
                    total_steps=7,
                    evaluation_steps=[3, 7],
                    data=data,
                    config=config,
                    device=torch.device("cpu"),
                )
            repeat = run_one(
                condition=condition,
                arm="absolute",
                seed=1,
                total_steps=7,
                evaluation_steps=[3, 7],
                data=data,
                config=config,
                device=torch.device("cpu"),
            )
        for arm, result in results.items():
            self.assertFalse(result["nonfinite"], arm)
            self.assertEqual([row["step"] for row in result["evaluations"]], [3, 7])
            self.assertEqual(result["evaluations"][-1]["windows_selected"], 28)
            self.assertEqual(len(result["diagnostics"]), 2)
            for row in result["diagnostics"]:
                self.assertLessEqual(
                    row["jvp_vs_per_sample_max_abs_error"], 1e-4 * row["jvp_max_abs"] + 1e-9
                )
        outcome = lambda run: [  # noqa: E731  (wall-clock fields excluded)
            (row["step"], row["validation_nll"], row["noise_windows_selected"])
            for row in run["evaluations"]
        ]
        self.assertEqual(outcome(results["absolute"]), outcome(repeat), "CPU runs must be deterministic")
        self.assertGreater(
            results["absolute"]["evaluations"][-1]["method_flops"],
            results["uniform"]["evaluations"][-1]["method_flops"],
        )
        # All arms train on the same first step, so their pools coincide.
        first_steps = {arm: result["evaluations"][0]["tokens_trained"] for arm, result in results.items()}
        self.assertEqual(len(set(first_steps.values())), 1)


class AnalysisTests(unittest.TestCase):
    """The registered decision rules on fabricated run files."""

    def setUp(self) -> None:
        sys.path.insert(0, str(EXPERIMENT))
        import analyze

        self.analyze = analyze
        self.config = json.loads((CONFIG_DIR / "fixed.json").read_text(encoding="utf-8"))

    def payloads(self, final: dict[str, float], uniform_long: float, spread: float = 0.01) -> dict:
        steps = self.config["evaluation"]["steps"]
        extra = self.config["evaluation"]["uniform_extra_steps"]
        payloads = {}
        for condition in ("clean", "noisy"):
            for arm in self.config["arms"]:
                for index, seed in enumerate(self.config["seeds"]):
                    offset = spread * (index - 2) / 2
                    arm_steps = steps + (extra if arm == "uniform" else [])
                    evaluations = []
                    for step in arm_steps:
                        value = final[arm] + offset + (6000 - min(step, 6000)) * 1e-4
                        if step > 6000:
                            value = uniform_long + offset
                        evaluations.append(
                            {
                                "step": step,
                                "validation_nll": value,
                                "noise_windows_selected": 4 * step if arm == "absolute" else 2 * step,
                                "windows_selected": 16 * step,
                                "method_seconds": 1.0,
                            }
                        )
                    payloads[(condition, arm, seed)] = {
                        "config_sha256": "x",
                        "git_commit": "y",
                        "dataset": {"unigram_validation_nll": 5.0},
                        "run": {
                            "condition": condition,
                            "arm": arm,
                            "seed": seed,
                            "evaluations": evaluations,
                            "diagnostics": [
                                {
                                    "step": 250,
                                    "jvp_max_abs": 1.0,
                                    "jvp_vs_per_sample_max_abs_error": 1e-6,
                                    "spearman": {"absolute|loss": 0.5},
                                }
                            ],
                            "nonfinite": False,
                        },
                    }
        return payloads

    def test_supported_when_every_primary_interval_is_negative(self) -> None:
        final = {"uniform": 2.5, "loss": 2.5, "gradnorm": 2.5, "cosine": 2.5, "signed": 2.45, "absolute": 2.4}
        result = self.analyze.analyze(self.payloads(final, uniform_long=2.45, spread=0.0), self.config)
        self.assertTrue(result["gates"]["all"])
        self.assertTrue(result["decision"].startswith("supported"))
        self.assertAlmostEqual(result["clean"]["noise_share_selected_at_primary_step"]["absolute"]["mean"], 0.25)

    def test_signal_without_efficiency(self) -> None:
        final = {"uniform": 2.5, "loss": 2.5, "gradnorm": 2.5, "cosine": 2.5, "signed": 2.45, "absolute": 2.4}
        result = self.analyze.analyze(self.payloads(final, uniform_long=2.2), self.config)
        self.assertTrue(result["decision"].startswith("selection signal supported"))

    def test_not_supported_when_a_comparator_ties(self) -> None:
        final = {"uniform": 2.5, "loss": 2.4, "gradnorm": 2.5, "cosine": 2.5, "signed": 2.45, "absolute": 2.4}
        result = self.analyze.analyze(self.payloads(final, uniform_long=2.2), self.config)
        self.assertTrue(result["decision"].startswith("not supported"))

    def test_capability_gate_invalidates(self) -> None:
        final = {arm: 4.5 for arm in self.config["arms"]}
        result = self.analyze.analyze(self.payloads(final, uniform_long=4.4), self.config)
        self.assertFalse(result["gates"]["uniform_beats_unigram"])
        self.assertTrue(result["decision"].startswith("invalid"))

    def test_incomplete_grid_is_reported(self) -> None:
        final = {arm: 2.5 for arm in self.config["arms"]}
        payloads = self.payloads(final, uniform_long=2.4)
        payloads.pop(("clean", "absolute", 8601))
        result = self.analyze.analyze(payloads, self.config)
        self.assertTrue(result["decision"].startswith("incomplete"))

    def test_t_table_matches_scipy_when_available(self) -> None:
        self.assertAlmostEqual(self.analyze.T_975[4], 2.7764451051977987)
        self.assertAlmostEqual(self.analyze.t_critical(4), 2.7764451051977987, places=9)


if __name__ == "__main__":
    unittest.main()
