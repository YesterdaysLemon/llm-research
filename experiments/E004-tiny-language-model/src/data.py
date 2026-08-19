from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


TOKEN_PATTERN = re.compile(r"<[^>\s]+>|[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.I)
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
NAMES = ("ava", "ben", "cora", "dax")
MODULUS = 11
OPERATIONS = (
    "add:1",
    "add:3",
    "multiply:2",
    "multiply:3",
    "subtract:2",
    "negate:0",
)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    @property
    def stoi(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode_tokens(self, values: Iterable[str]) -> list[int]:
        lookup = self.stoi
        unknown = lookup["<unk>"]
        return [lookup.get(value.lower(), unknown) for value in values]

    def encode(self, text: str) -> list[int]:
        return self.encode_tokens(tokenize(text))

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.tokens[int(index)] for index in ids]


def controlled_required_tokens() -> tuple[str, ...]:
    text = " ".join(
        [
            "<bos> <eos> starts at applies add multiply subtract negate where is",
            " ".join(NAMES),
            " ".join(str(index) for index in range(MODULUS)),
            ". ?",
        ]
    )
    return tuple(dict.fromkeys(tokenize(text)))


def build_vocabulary(stories: Sequence[str], max_size: int) -> Vocabulary:
    if max_size < len(SPECIAL_TOKENS) + len(controlled_required_tokens()):
        raise ValueError("vocabulary too small for required controlled tokens")
    counts: Counter[str] = Counter()
    for story in stories:
        counts.update(tokenize(story))
    ordered: list[str] = list(SPECIAL_TOKENS)
    for token in controlled_required_tokens():
        if token not in ordered:
            ordered.append(token)
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if token not in ordered:
            ordered.append(token)
        if len(ordered) == max_size:
            break
    return Vocabulary(tuple(ordered))


def load_story_split(
    path: Path, *, train_stories: int, validation_stories: int
) -> tuple[list[str], list[str]]:
    text = path.read_text(encoding="utf-8")
    stories = [item.strip() for item in text.split("<|endoftext|>") if item.strip()]
    required = train_stories + validation_stories
    if len(stories) < required:
        raise ValueError(f"need {required} stories, found {len(stories)}")
    return stories[:train_stories], stories[train_stories:required]


def pack_stories(stories: Sequence[str], vocabulary: Vocabulary) -> torch.Tensor:
    lookup = vocabulary.stoi
    packed: list[int] = []
    for story in stories:
        packed.append(lookup["<bos>"])
        packed.extend(vocabulary.encode(story))
        packed.append(lookup["<eos>"])
    return torch.tensor(packed, dtype=torch.long)


def apply_operations(start: int, operations: Sequence[str]) -> int:
    state = int(start)
    for operation in operations:
        kind, amount_text = operation.split(":")
        amount = int(amount_text)
        if kind == "add":
            state += amount
        elif kind == "multiply":
            state *= amount
        elif kind == "subtract":
            state -= amount
        elif kind == "negate":
            state = -state
        else:
            raise ValueError(f"unknown operation: {operation}")
        state %= MODULUS
    return state


def forbidden_occurrences(
    operations: Sequence[str], forbidden_pairs: set[tuple[str, str]]
) -> list[tuple[str, str]]:
    return [
        (left, right)
        for left, right in zip(operations, operations[1:])
        if (left, right) in forbidden_pairs
    ]


def contains_forbidden_pair(
    operations: Sequence[str], forbidden_pairs: set[tuple[str, str]]
) -> bool:
    return bool(forbidden_occurrences(operations, forbidden_pairs))


def controlled_record(name: str, start: int, operations: Sequence[str]) -> str:
    clauses = " ".join(
        f"{name} applies {operation.split(':')[0]} {operation.split(':')[1]} ."
        for operation in operations
    )
    answer = apply_operations(start, operations)
    return (
        f"<bos> {name} starts at {start} . {clauses} "
        f"where is {name} ? {answer} . <eos>"
    )


def generate_controlled_records(
    *,
    count: int,
    depths: Sequence[int],
    seed: int,
    forbidden_pairs: Sequence[Sequence[str]],
) -> list[str]:
    import random

    rng = random.Random(seed)
    forbidden = {tuple(pair) for pair in forbidden_pairs}
    records: list[str] = []
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not sample enough controlled records")
        depth = int(rng.choice(tuple(depths)))
        operations = [rng.choice(OPERATIONS) for _ in range(depth)]
        if contains_forbidden_pair(operations, forbidden):
            continue
        records.append(
            controlled_record(rng.choice(NAMES), rng.randrange(MODULUS), operations)
        )
    return records


def generate_eval_records(
    *,
    count: int,
    depths: Sequence[int],
    seed: int,
    forbidden_pairs: Sequence[Sequence[str]],
    require_heldout: bool,
) -> list[tuple[int, str, str, int]]:
    """Return depth, pair label, prompt, and answer with a balanced frozen grid."""
    import random

    rng = random.Random(seed)
    ordered_heldout = tuple(tuple(pair) for pair in forbidden_pairs)
    forbidden = set(ordered_heldout)
    ordered_depths = tuple(int(depth) for depth in depths)
    if require_heldout and any(depth < 2 for depth in ordered_depths):
        raise ValueError("held-out compositions require depth at least two")
    rows: list[tuple[int, str, str, int]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError("could not sample controlled evaluation records")
        row_index = len(rows)
        depth = ordered_depths[row_index % len(ordered_depths)]
        operations = [rng.choice(OPERATIONS) for _ in range(depth)]
        if contains_forbidden_pair(operations, forbidden):
            continue
        pair_label = "none"
        if require_heldout:
            pair_index = (row_index // len(ordered_depths)) % len(ordered_heldout)
            pair = ordered_heldout[pair_index]
            insertion = rng.randrange(depth - 1)
            operations[insertion : insertion + 2] = pair
            occurrences = forbidden_occurrences(operations, forbidden)
            if occurrences != [pair]:
                continue
            pair_label = f"{pair[0]}->{pair[1]}"
        name = rng.choice(NAMES)
        start = rng.randrange(MODULUS)
        full = controlled_record(name, start, operations)
        prompt = full.rsplit(" ", 3)[0]
        rows.append((depth, pair_label, prompt, apply_operations(start, operations)))
    return rows


def encode_controlled_records(
    records: Sequence[str], vocabulary: Vocabulary
) -> list[torch.Tensor]:
    return [torch.tensor(vocabulary.encode(record), dtype=torch.long) for record in records]


class PairedBatchStream:
    """Deterministic natural windows plus complete, boundary-aligned controlled records."""

    def __init__(
        self,
        natural: torch.Tensor,
        controlled_records: Sequence[torch.Tensor],
        *,
        sequence_length: int,
        controlled_fraction: float,
        pad_id: int,
        question_id: int,
        seed: int,
    ) -> None:
        if natural.numel() <= sequence_length:
            raise ValueError("natural stream is shorter than one sequence")
        if not controlled_records:
            raise ValueError("controlled record list is empty")
        if any(record.numel() > sequence_length + 1 for record in controlled_records):
            raise ValueError("a controlled record exceeds one sequence")
        self.natural = natural
        self.controlled_records = tuple(controlled_records)
        self.sequence_length = int(sequence_length)
        self.controlled_fraction = float(controlled_fraction)
        self.pad_id = int(pad_id)
        self.question_id = int(question_id)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.controlled_carry = 0.0

    def _controlled_row(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        start = int(
            torch.randint(
                len(self.controlled_records), (), generator=self.generator
            ).item()
        )
        values: list[int] = []
        records = 0
        for offset in range(len(self.controlled_records)):
            record = self.controlled_records[
                (start + offset) % len(self.controlled_records)
            ]
            record_values = record.tolist()
            if len(values) + len(record_values) > self.sequence_length + 1:
                break
            values.extend(record_values)
            records += 1
        valid_length = len(values)
        if records == 0 or valid_length < 2:
            raise RuntimeError("could not pack a controlled row")
        values.extend(
            [self.pad_id] * (self.sequence_length + 1 - valid_length)
        )
        row = torch.tensor(values, dtype=torch.long)
        mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        mask[: valid_length - 1] = True
        answer_mask = row[:-1].eq(self.question_id) & mask
        if int(answer_mask.sum().item()) != records:
            raise RuntimeError("controlled answer mask does not match packed records")
        return row, mask, answer_mask, records

    def batch(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]
    ]:
        target_controlled = (
            self.controlled_carry + int(batch_size) * self.controlled_fraction
        )
        controlled_count = int(target_controlled)
        self.controlled_carry = target_controlled - controlled_count
        source_flags = torch.tensor(
            [1] * controlled_count + [0] * (int(batch_size) - controlled_count),
            dtype=torch.long,
        )
        permutation = torch.randperm(int(batch_size), generator=self.generator)
        source_flags = source_flags[permutation]
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        answer_masks: list[torch.Tensor] = []
        controlled_records = 0
        source_tokens: defaultdict[str, int] = defaultdict(int)
        for flag in source_flags.tolist():
            if flag:
                row, mask, answer_mask, records = self._controlled_row()
                controlled_records += records
                source_tokens["controlled"] += int(mask.sum().item())
                source_tokens["controlled_answers"] += int(
                    answer_mask.sum().item()
                )
            else:
                maximum = self.natural.numel() - self.sequence_length - 1
                offset = int(
                    torch.randint(
                        maximum + 1, (), generator=self.generator
                    ).item()
                )
                row = self.natural[offset : offset + self.sequence_length + 1]
                mask = torch.ones(self.sequence_length, dtype=torch.bool)
                answer_mask = torch.zeros(self.sequence_length, dtype=torch.bool)
                source_tokens["natural"] += self.sequence_length
            rows.append(row)
            masks.append(mask)
            answer_masks.append(answer_mask)
        tokens = torch.stack(rows)
        metadata = {
            "natural_sequences": int(batch_size) - controlled_count,
            "controlled_sequences": controlled_count,
            "controlled_records": controlled_records,
            "natural_supervised_tokens": source_tokens["natural"],
            "controlled_supervised_tokens": source_tokens["controlled"],
            "controlled_answer_tokens": source_tokens["controlled_answers"],
        }
        return (
            tokens[:, :-1].contiguous(),
            tokens[:, 1:].contiguous(),
            torch.stack(masks),
            torch.stack(answer_masks),
            metadata,
        )
