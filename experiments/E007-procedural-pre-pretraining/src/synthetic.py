"""Synthetic pre-pretraining sources for E007.

Two sources, both adapted from arXiv:2609.30063:

* `pcfg`: random probabilistic context-free grammars (paper App. H), one fresh
  grammar per row, terminals drawn from the natural vocabulary's content ids.
* `programs`: programs sampled i.i.d. from the uniform prior over the paper's
  Brainf*ck-like alphabet with its ten macros (App. E, Table 4), executed on a
  circular byte tape; output bytes are mapped to content ids by a fixed
  injection.

All randomness comes from `torch.Generator` blocks so rows are identical across
platforms and Python versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

PRIMITIVES = "<>+-[].,"
MACROS = {
    "Z": "[-]",
    "R": "[->+<]",
    "L": "[->+++<]",
    "N": "[-<->]",
    "C": "[->+>+<<]",
    "G": "[>]",
    "H": "[<]",
    "W": "[[-]>+<]",
    "V": "[.>]",
    "X": "[-]" + "+" * 16,
}
END = "F"
ALPHABET = PRIMITIVES + "".join(MACROS) + END  # 19 symbols


class RandomStream:
    """Uniform floats from a seeded torch generator, served in blocks."""

    def __init__(self, seed: int, block: int = 65536) -> None:
        self.generator = torch.Generator().manual_seed(int(seed))
        self.block = int(block)
        self.values: list[float] = []
        self.position = 0

    def uniform(self) -> float:
        if self.position >= len(self.values):
            self.values = torch.rand(self.block, generator=self.generator, dtype=torch.float64).tolist()
            self.position = 0
        value = self.values[self.position]
        self.position += 1
        return value

    def integer(self, low: int, high: int) -> int:
        """Uniform integer in [low, high] inclusive."""
        return low + min(int(self.uniform() * (high - low + 1)), high - low)

    def choice(self, values: list[Any] | tuple[Any, ...] | str) -> Any:
        return values[self.integer(0, len(values) - 1)]

    def sample_distinct(self, low: int, high: int, count: int) -> list[int]:
        """`count` distinct integers from [low, high) without replacement."""
        chosen: list[int] = []
        seen: set[int] = set()
        while len(chosen) < count:
            value = self.integer(low, high - 1)
            if value not in seen:
                seen.add(value)
                chosen.append(value)
        return chosen


# --------------------------------------------------------------------------
# Random PCFGs


@dataclass
class Grammar:
    terminals: list[int]
    productions: list[list[tuple[list[tuple[bool, int]], float]]]  # per nonterminal
    one_step_yield: list[list[int]]  # a terminal-only right-hand side per nonterminal


def sample_grammar(rng: RandomStream, params: dict[str, Any], content_low: int, content_high: int) -> Grammar:
    n_terminals = rng.integer(int(params["terminals_min"]), int(params["terminals_max"]))
    terminals = rng.sample_distinct(content_low, content_high, n_terminals)
    n_nonterminals = rng.integer(int(params["nonterminals_min"]), int(params["nonterminals_max"]))
    productions: list[list[tuple[list[tuple[bool, int]], float]]] = []
    for _ in range(n_nonterminals):
        rules: list[tuple[list[tuple[bool, int]], float]] = []
        for _ in range(rng.integer(int(params["productions_min"]), int(params["productions_max"]))):
            weight = rng.uniform() + 1e-6
            rhs: list[tuple[bool, int]] = []
            for _ in range(rng.integer(int(params["rhs_min"]), int(params["rhs_max"]))):
                if rng.uniform() < float(params["terminal_probability"]):
                    rhs.append((True, rng.choice(terminals)))
                else:
                    rhs.append((False, rng.integer(0, n_nonterminals - 1)))
            rules.append((rhs, weight))
        total = sum(weight for _, weight in rules)
        productions.append([(rhs, weight / total) for rhs, weight in rules])
    # Productivity repair: every nonterminal can terminate in one step.
    one_step: list[list[int]] = []
    for index, rules in enumerate(productions):
        terminal_only = [rhs for rhs, _ in rules if all(is_terminal for is_terminal, _ in rhs)]
        if not terminal_only:
            replace = rng.integer(0, len(rules) - 1)
            length = rng.integer(int(params["rhs_min"]), int(params["rhs_max"]))
            fresh = [(True, rng.choice(terminals)) for _ in range(length)]
            rules[replace] = (fresh, rules[replace][1])
            terminal_only = [fresh]
        one_step.append([symbol for _, symbol in terminal_only[0]])
    return Grammar(terminals, productions, one_step)


def derive_word(rng: RandomStream, grammar: Grammar, params: dict[str, Any]) -> list[int]:
    """Leftmost derivation from the start symbol with an explicit stack."""
    stack: list[tuple[bool, int]] = [(False, 0)]
    output: list[int] = []
    expansions = 0
    max_length = int(params["max_word_length"])
    max_expansions = int(params["max_expansions"])
    while stack and len(output) < max_length and expansions < max_expansions:
        is_terminal, symbol = stack.pop()
        if is_terminal:
            output.append(symbol)
            continue
        expansions += 1
        draw = rng.uniform()
        cumulative = 0.0
        rules = grammar.productions[symbol]
        chosen = rules[-1][0]
        for rhs, probability in rules:
            cumulative += probability
            if draw < cumulative:
                chosen = rhs
                break
        stack.extend(reversed(chosen))
    if not output:
        output = list(grammar.one_step_yield[0])
    return output[:max_length]


def pcfg_row(rng: RandomStream, params: dict[str, Any], length: int, content_low: int, content_high: int) -> list[int]:
    """One row of `length` tokens: words from one fresh grammar, concatenated."""
    grammar = sample_grammar(rng, params, content_low, content_high)
    row: list[int] = []
    while len(row) < length:
        row.extend(derive_word(rng, grammar, params))
    return row[:length]


# --------------------------------------------------------------------------
# Uniform-prior programs


def expand_macros(program: str) -> str:
    return "".join(MACROS.get(symbol, symbol) for symbol in program if symbol != END)


def match_brackets(code: str) -> dict[int, int]:
    """Matched bracket pairs; unmatched brackets are absent and act as no-ops."""
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for index, symbol in enumerate(code):
        if symbol == "[":
            stack.append(index)
        elif symbol == "]" and stack:
            opening = stack.pop()
            pairs[opening] = index
            pairs[index] = opening
    return pairs


def execute(
    program: str,
    *,
    tape_input: RandomStream,
    max_outputs: int,
    step_budget: int,
    tape_cells: int,
) -> list[int]:
    """Run a program; stop at its end, the step budget, or `max_outputs` bytes."""
    code = expand_macros(program)
    pairs = match_brackets(code)
    tape = [0] * tape_cells
    head = 0
    pointer = 0
    steps = 0
    outputs: list[int] = []
    length = len(code)
    while pointer < length and steps < step_budget and len(outputs) < max_outputs:
        symbol = code[pointer]
        if symbol == "+":
            tape[head] = (tape[head] + 1) & 255
        elif symbol == "-":
            tape[head] = (tape[head] - 1) & 255
        elif symbol == ">":
            head = (head + 1) % tape_cells
        elif symbol == "<":
            head = (head - 1) % tape_cells
        elif symbol == ".":
            outputs.append(tape[head])
        elif symbol == ",":
            tape[head] = tape_input.integer(0, 255)
        elif symbol == "[":
            if tape[head] == 0 and pointer in pairs:
                pointer = pairs[pointer]
        elif symbol == "]":
            if tape[head] != 0 and pointer in pairs:
                pointer = pairs[pointer]
        pointer += 1
        steps += 1
    return outputs


def sample_program(rng: RandomStream, max_tokens: int) -> str:
    """Tokens i.i.d. uniform over the 19-symbol alphabet until F (inclusive)."""
    symbols: list[str] = []
    while len(symbols) < max_tokens:
        symbol = rng.choice(ALPHABET)
        symbols.append(symbol)
        if symbol == END:
            break
    return "".join(symbols)


def byte_token_map(seed: int, content_low: int, content_high: int) -> list[int]:
    """A fixed injection from byte values 0..255 into content token ids."""
    if content_high - content_low < 256:
        raise ValueError("the vocabulary needs at least 256 content ids to map program bytes")
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(content_high - content_low, generator=generator)[:256]
    return [content_low + int(value) for value in permutation.tolist()]


def accepted_output(rng: RandomStream, params: dict[str, Any], length: int) -> list[int]:
    """Byte outputs of the next uniform-prior program that emits at least `min_outputs` bytes."""
    while True:
        program = sample_program(rng, int(params["max_program_tokens"]))
        outputs = execute(
            program,
            tape_input=rng,
            max_outputs=length,
            step_budget=int(params["step_budget"]),
            tape_cells=int(params["tape_cells"]),
        )
        if len(outputs) >= int(params["min_outputs"]):
            return outputs


def _bank_chunk(arguments: tuple[dict[str, Any], int, int, int]) -> list[list[int]]:
    params, seed, count, length = arguments
    rng = RandomStream(seed)
    return [accepted_output(rng, params, length) for _ in range(count)]


def build_program_bank(
    params: dict[str, Any], *, length: int, processes: int = 1
) -> list[list[int]]:
    """A fixed bank of accepted program outputs, identical for any process count.

    Chunk `c` is generated from seed `bank_seed + c`, so the bank does not
    depend on how chunks are scheduled across processes.
    """
    size = int(params["bank_size"])
    chunk = int(params["bank_chunk"])
    jobs = [
        (params, int(params["bank_seed"]) + index, min(chunk, size - start), int(length))
        for index, start in enumerate(range(0, size, chunk))
    ]
    if processes > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(processes) as pool:
            chunks = pool.map(_bank_chunk, jobs)
    else:
        chunks = [_bank_chunk(job) for job in jobs]
    return [row for rows in chunks for row in rows]


# --------------------------------------------------------------------------
# Batches


class SyntheticStream:
    """Deterministic batches of (inputs, targets, target mask) for one source.

    `pcfg` rows are generated fresh. `programs` rows come from a fixed bank of
    accepted program outputs, visited in a seed-specific order without
    replacement and reshuffled when exhausted.
    """

    def __init__(
        self,
        source: str,
        params: dict[str, Any],
        *,
        seed: int,
        sequence_length: int,
        content_low: int,
        content_high: int,
        pad_id: int,
        mapping_seed: int,
        bank: list[list[int]] | None = None,
    ) -> None:
        if source not in ("pcfg", "programs"):
            raise ValueError(f"unknown synthetic source {source!r}")
        if source == "programs" and not bank:
            raise ValueError("the programs source needs a program bank")
        self.source = source
        self.params = params
        self.rng = RandomStream(seed)
        self.order = torch.Generator().manual_seed(int(seed))
        self.length = int(sequence_length) + 1
        self.content_low = int(content_low)
        self.content_high = int(content_high)
        self.pad_id = int(pad_id)
        self.mapping = byte_token_map(mapping_seed, content_low, content_high)
        self.bank = bank
        self.queue: list[int] = []

    def row(self) -> tuple[list[int], int]:
        if self.source == "pcfg":
            return pcfg_row(self.rng, self.params, self.length, self.content_low, self.content_high), self.length
        if not self.queue:
            self.queue = torch.randperm(len(self.bank), generator=self.order).tolist()[::-1]
        outputs = self.bank[self.queue.pop()][: self.length]
        return [self.mapping[value] for value in outputs], len(outputs)

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.full((int(size), self.length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((int(size), self.length - 1), dtype=torch.bool)
        for index in range(int(size)):
            values, valid = self.row()
            rows[index, :valid] = torch.tensor(values[:valid], dtype=torch.long)
            mask[index, : valid - 1] = True
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous(), mask
