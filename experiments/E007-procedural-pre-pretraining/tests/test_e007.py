from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))
sys.path.insert(0, str(EXPERIMENT))

import analyze  # noqa: E402
from natural import validation_windows  # noqa: E402
from probes import evaluate_probes, probe_prompts  # noqa: E402
from synthetic import (  # noqa: E402
    ALPHABET,
    MACROS,
    RandomStream,
    SyntheticStream,
    build_program_bank,
    byte_token_map,
    execute,
    expand_macros,
    pcfg_row,
    sample_grammar,
    sample_program,
)
from train import (  # noqa: E402
    ARMS,
    run_arm,
    shuffle_within_tensors,
    state_sha256,
    validate_config,
)
from model import TinyCausalLM  # noqa: E402  (E004)

CONFIG_DIR = EXPERIMENT / "config"


class FixedTape:
    def __init__(self, values: list[int]) -> None:
        self.values = list(values)

    def integer(self, low: int, high: int) -> int:
        return self.values.pop(0)


def run(program: str, tape: list[int] | None = None, outputs: int = 12, budget: int = 100000) -> list[int]:
    return execute(program, tape_input=FixedTape(tape or []), max_outputs=outputs, step_budget=budget, tape_cells=256)


def load(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


class InterpreterTests(unittest.TestCase):
    def test_reproduces_the_papers_discovered_sequences(self) -> None:
        # arXiv:2609.30063 Table 1 (programs without the leading "S" prefix byte).
        self.assertEqual(run("+[.++]")[:5], [1, 3, 5, 7, 9])
        self.assertEqual(run("+[.L>]")[:5], [1, 3, 9, 27, 81])
        self.assertEqual(run(",[[.C>.C>]", [1])[:8], [1, 1, 2, 3, 5, 8, 13, 21])
        self.assertEqual(run(",.[<C>>VX<RX++]", [9])[:5], [9, 25, 59, 111, 181])
        self.assertEqual(run("+[[-.L>L>-]-]")[:5], [0, 254, 236, 74, 152])

    def test_macro_table(self) -> None:
        self.assertEqual(len(ALPHABET), 19)
        self.assertEqual(MACROS["X"], "[-]" + "+" * 16)
        self.assertEqual(expand_macros("ZRF"), "[-][->+<]")

    def test_three_iteration_loop_from_the_paper(self) -> None:
        self.assertEqual(run("+++[>+.<-]F"), [1, 2, 3])

    def test_unmatched_brackets_are_no_ops_and_tape_wraps(self) -> None:
        self.assertEqual(run("]+.[+."), [1, 2])
        self.assertEqual(run("-."), [255])
        self.assertEqual(run("<+.>."), [1, 0])

    def test_step_budget_and_output_limit(self) -> None:
        self.assertEqual(run("+[]", budget=50), [])
        self.assertEqual(len(run("+[.]", outputs=7)), 7)

    def test_uniform_prior_programs_end_with_the_end_token(self) -> None:
        rng = RandomStream(4)
        for _ in range(200):
            program = sample_program(rng, 256)
            self.assertTrue(set(program) <= set(ALPHABET))
            self.assertEqual(program[-1], "F")


class GrammarTests(unittest.TestCase):
    params = load("fixed.json")["synthetic"]["sources"]["pcfg"]

    def test_grammars_are_productive(self) -> None:
        rng = RandomStream(1)
        for _ in range(100):
            grammar = sample_grammar(rng, self.params, 4, 2048)
            self.assertTrue(2 <= len(grammar.terminals) <= 16)
            for rules in grammar.productions:
                self.assertTrue(any(all(t for t, _ in rhs) for rhs, _ in rules))
                self.assertAlmostEqual(sum(p for _, p in rules), 1.0)

    def test_rows_are_full_and_use_few_terminals(self) -> None:
        rng = RandomStream(2)
        for _ in range(100):
            row = pcfg_row(rng, self.params, 65, 4, 2048)
            self.assertEqual(len(row), 65)
            self.assertTrue(all(4 <= token < 2048 for token in row))
            self.assertLessEqual(len(set(row)), 16)

    def test_rows_are_deterministic(self) -> None:
        a = [pcfg_row(RandomStream(3), self.params, 65, 4, 2048) for _ in range(1)]
        b = [pcfg_row(RandomStream(3), self.params, 65, 4, 2048) for _ in range(1)]
        self.assertEqual(a, b)


class BankTests(unittest.TestCase):
    params = {**load("fixed.json")["synthetic"]["sources"]["programs"], "bank_size": 12, "bank_chunk": 5}

    def test_bank_is_independent_of_process_count(self) -> None:
        serial = build_program_bank(self.params, length=65, processes=1)
        parallel = build_program_bank(self.params, length=65, processes=2)
        self.assertEqual(serial, parallel)
        self.assertEqual(len(serial), 12)
        self.assertTrue(all(len(row) == 65 for row in serial))

    def test_byte_map_is_an_injection_into_content_ids(self) -> None:
        mapping = byte_token_map(8799, 4, 2048)
        self.assertEqual(len(set(mapping)), 256)
        self.assertTrue(all(4 <= token < 2048 for token in mapping))
        self.assertEqual(mapping, byte_token_map(8799, 4, 2048))

    def test_stream_visits_each_bank_row_once_per_epoch(self) -> None:
        bank = [[index] * 65 for index in range(10)]
        stream = SyntheticStream(
            "programs", self.params, seed=1, sequence_length=64, content_low=4, content_high=2048,
            pad_id=0, mapping_seed=8799, bank=bank,
        )
        inputs, _, mask = stream.batch(10)
        mapping = byte_token_map(8799, 4, 2048)
        seen = sorted(mapping.index(int(row[0])) for row in inputs)
        self.assertEqual(seen, list(range(10)))
        self.assertTrue(bool(mask.all()))


class ControlTests(unittest.TestCase):
    def test_shuffle_preserves_each_tensor_value_multiset_and_tying(self) -> None:
        torch.manual_seed(0)
        model = TinyCausalLM(vocab_size=30, sequence_length=8, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0)
        before = {name: value.detach().flatten().sort().values.clone() for name, value in model.named_parameters()}
        digest = state_sha256(model)
        shuffle_within_tensors(model, 5)
        self.assertNotEqual(state_sha256(model), digest)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value.detach().flatten().sort().values, before[name])
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)


class ProbeTests(unittest.TestCase):
    def test_prompt_structure(self) -> None:
        prompts = probe_prompts(load("fixed.json")["probes"])
        tokens, positions, expected = prompts["copy"][0]
        self.assertEqual(tokens[:20], tokens[20:])
        self.assertEqual([tokens[p + 1] for p in positions], expected)
        tokens, positions, expected = prompts["recall"][0]
        keys, values = tokens[0:-1:2], tokens[1:-1:2]
        self.assertEqual(expected, [values[keys.index(tokens[-1])]])
        self.assertEqual(len(prompts["copy"]), 512)

    def test_perfect_copier_scores_one(self) -> None:
        class Copier(torch.nn.Module):
            def forward(self, tokens):
                logits = torch.full((*tokens.shape, 1200), -1e9)
                for row in range(tokens.shape[0]):
                    for position in range(tokens.shape[1]):
                        earlier = (tokens[row, :position] == tokens[row, position]).nonzero()
                        if len(earlier):
                            logits[row, position, tokens[row, int(earlier[0]) + 1]] = 0.0
                return logits, []

        prompts = probe_prompts({**load("fixed.json")["probes"], "trials": 8})
        scores = evaluate_probes(Copier(), prompts, device=torch.device("cpu"))
        self.assertEqual(scores["copy"], 1.0)
        self.assertEqual(scores["recall"], 1.0)


class ConfigTests(unittest.TestCase):
    def test_fixed_config(self) -> None:
        config = load("fixed.json")
        validate_config(config)
        self.assertEqual(config["arms"], list(ARMS))
        self.assertEqual(config["seeds"], [8701, 8702, 8703, 8704, 8705])
        self.assertEqual(config["synthetic"]["steps"], 8000)
        self.assertEqual(config["natural"]["warm_steps"], 8000)
        self.assertEqual(config["natural"]["scratch_steps"], 16000)
        self.assertEqual(
            config["synthetic"]["sources"]["programs"]["bank_size"],
            config["synthetic"]["steps"] * config["training"]["batch_size"],
        )

    def test_smoke_shares_every_design_section(self) -> None:
        fixed, smoke = load("fixed.json"), load("smoke.json")
        validate_config(smoke)
        for key in ("dataset", "student", "training", "arms", "natural", "probes", "gates", "shuffle_seed_offset"):
            if key == "natural":
                self.assertEqual(fixed[key]["stream_seed_offset"], smoke[key]["stream_seed_offset"])
                continue
            self.assertEqual(fixed[key], smoke[key], key)
        self.assertEqual(fixed["synthetic"]["sources"], smoke["synthetic"]["sources"])
        self.assertNotIn(smoke["seeds"][0], fixed["seeds"])

    def test_dataset_section_matches_e006_and_e004(self) -> None:
        e006 = json.loads((EXPERIMENT.parent / "E006-learning-progress-selection" / "config" / "fixed.json").read_text())
        self.assertEqual(load("fixed.json")["dataset"], e006["dataset"])


class EndToEndTests(unittest.TestCase):
    """Every arm runs on synthetic 'natural' tokens with a tiny model; checks code paths only."""

    def config(self) -> dict:
        config = load("smoke.json")
        config["student"] = {**config["student"], "d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32}
        config["dataset"] = {**config["dataset"], "sequence_length": 64}
        config["training"] = {**config["training"], "batch_size": 4, "use_bfloat16": False}
        config["synthetic"] = {**config["synthetic"], "steps": 6, "evaluation_steps": [3, 6], "heldout_rows": 8}
        config["synthetic"]["sources"] = {
            **config["synthetic"]["sources"],
            "programs": {**config["synthetic"]["sources"]["programs"], "bank_size": 10, "bank_chunk": 5},
        }
        config["natural"] = {**config["natural"], "warm_steps": 6, "scratch_steps": 12}
        config["evaluation"] = {"batch_size": 16, "interval": 3}
        config["probes"] = {**config["probes"], "trials": 4}
        return config

    def data(self) -> dict:
        class Vocabulary:
            tokens = tuple(f"t{index}" for index in range(1200))

        generator = torch.Generator().manual_seed(5)
        return {
            "vocabulary": Vocabulary(),
            "train": torch.randint(4, 1200, (4000,), generator=generator),
            "validation_windows": validation_windows(torch.randint(4, 1200, (641,), generator=generator), 64),
        }

    def test_all_arms_run_deterministically_and_reuse_checkpoints(self) -> None:
        config, data = self.config(), self.data()
        bank = build_program_bank(config["synthetic"]["sources"]["programs"], length=65, processes=1)
        prompts = probe_prompts(config["probes"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory)
            results = {
                arm: run_arm(
                    arm, seed=1, data=data, config=config, prompts=prompts,
                    device=torch.device("cpu"), checkpoint_dir=checkpoints, bank=bank,
                )
                for arm in ARMS
            }
            self.assertEqual(len(list(checkpoints.glob("*.pt"))), 2)
            again = run_arm(
                "pcfg", seed=1, data=data, config=config, prompts=prompts,
                device=torch.device("cpu"), checkpoint_dir=checkpoints, bank=bank,
            )
        steps = [row["step"] for row in results["scratch"]["natural"]["evaluations"]]
        self.assertEqual(steps, [0, 3, 6, 9, 12])
        self.assertEqual([row["step"] for row in results["pcfg"]["natural"]["evaluations"]], [0, 3, 6])
        self.assertEqual(results["pcfg"]["natural"]["evaluations"], again["natural"]["evaluations"])
        self.assertEqual(
            results["pcfg"]["synthetic"]["state_sha256"], results["pcfg_shuffled"]["synthetic"]["state_sha256"]
        )
        self.assertNotEqual(
            results["pcfg"]["natural_start_state_sha256"], results["pcfg_shuffled"]["natural_start_state_sha256"]
        )
        self.assertEqual(set(results["scratch"]["natural"]["probes"]), {"0", "6", "12"})
        for arm, result in results.items():
            self.assertFalse(result["natural"]["nonfinite"], arm)


class AnalysisTests(unittest.TestCase):
    config = load("fixed.json")

    def payloads(self, final: dict[str, float], matched: float, learned: float = 3.0) -> dict:
        payloads = {}
        for arm in self.config["arms"]:
            for index, seed in enumerate(self.config["seeds"]):
                offset = 0.002 * (index - 2)
                last = 16000 if arm == "scratch" else 8000
                evaluations = []
                for step in range(0, last + 1, 250):
                    value = final[arm] + offset + (8000 - min(step, 8000)) * 1e-4
                    if step > 8000:
                        value = matched + offset
                    evaluations.append({"step": step, "validation_nll": value})
                synthetic = None
                if arm != "scratch":
                    synthetic = {"heldout_curve": [{"heldout_nll": 7.6}, {"heldout_nll": 7.6 - learned}]}
                payloads[(arm, seed)] = {
                    "config_sha256": "x",
                    "git_commit": "y",
                    "dataset": {"unigram_validation_nll": 5.3},
                    "program_bank": {"sha256": "z"},
                    "run": {
                        "arm": arm,
                        "seed": seed,
                        "synthetic": synthetic,
                        "natural": {"evaluations": evaluations, "probes": {"0": {"copy": 0.1}}, "nonfinite": False},
                    },
                }
        return payloads

    def test_supported(self) -> None:
        final = {"scratch": 2.5, "pcfg": 2.3, "pcfg_shuffled": 2.5, "programs": 2.5, "programs_shuffled": 2.5}
        result = analyze.analyze(self.payloads(final, matched=2.4), self.config)
        self.assertTrue(result["decision"].startswith("supported: pcfg"), result["decision"])

    def test_head_start_without_compute_efficiency(self) -> None:
        final = {"scratch": 2.5, "pcfg": 2.4, "pcfg_shuffled": 2.5, "programs": 2.5, "programs_shuffled": 2.5}
        result = analyze.analyze(self.payloads(final, matched=2.2), self.config)
        self.assertTrue(result["decision"].startswith("head start supported for pcfg"), result["decision"])
        self.assertAlmostEqual(result["natural_token_savings_to_scratch_level"]["pcfg"]["8000"]["mean_where_reached"], 1 - 7000 / 8000, places=6)

    def test_statistics_only(self) -> None:
        final = {"scratch": 2.5, "pcfg": 2.4, "pcfg_shuffled": 2.4, "programs": 2.5, "programs_shuffled": 2.5}
        result = analyze.analyze(self.payloads(final, matched=2.2), self.config)
        self.assertTrue(result["decision"].startswith("not supported as procedural structure"), result["decision"])

    def test_synthetic_gate(self) -> None:
        final = {arm: 2.5 for arm in self.config["arms"]}
        result = analyze.analyze(self.payloads(final, matched=2.2, learned=0.1), self.config)
        self.assertTrue(result["decision"].startswith("invalid: no synthetic stage"), result["decision"])

    def test_t_table_matches_scipy(self) -> None:
        for confidence, table in analyze.T_TABLE.items():
            for df, value in table.items():
                self.assertAlmostEqual(analyze.t_critical(confidence, df), value, places=6)


if __name__ == "__main__":
    unittest.main()
