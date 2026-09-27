"""TinyStories natural stage for E007, reproducing E004's tokenizer and split."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

E004_SRC = Path(__file__).resolve().parents[2] / "E004-tiny-language-model" / "src"
if str(E004_SRC) not in sys.path:
    sys.path.insert(0, str(E004_SRC))

from data import (  # noqa: E402  (E004, frozen)
    SPECIAL_TOKENS,
    build_vocabulary,
    load_story_split,
    pack_stories,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]


def vocabulary_sha256(tokens: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def prepare_natural_data(dataset: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / dataset["path"]
    if not source.exists():
        raise FileNotFoundError(
            "missing dataset; run experiments/E004-tiny-language-model/src/fetch_data.py: "
            f"{source}"
        )
    if source.stat().st_size != int(dataset["bytes"]):
        raise RuntimeError("dataset byte count does not match frozen config")
    source_hash = sha256_file(source)
    if source_hash.lower() != str(dataset["sha256"]).lower():
        raise RuntimeError("dataset SHA-256 does not match frozen config")
    train_stories, validation_stories = load_story_split(
        source,
        train_stories=int(dataset["train_stories"]),
        validation_stories=int(dataset["validation_stories"]),
    )
    vocabulary = build_vocabulary(train_stories, int(dataset["vocab_size"]))
    digest = vocabulary_sha256(vocabulary.tokens)
    expected = dataset.get("expected_vocabulary_sha256")
    if expected is not None and digest != expected:
        raise RuntimeError(f"vocabulary SHA-256 {digest} does not reproduce E004 ({expected})")
    train = pack_stories(train_stories, vocabulary)
    validation = pack_stories(validation_stories, vocabulary)
    for key, stream in (("train", train), ("validation", validation)):
        expected_tokens = dataset.get(f"expected_{key}_tokens")
        if expected_tokens is not None and stream.numel() != int(expected_tokens):
            raise RuntimeError(f"{key} token count {stream.numel()} does not reproduce E004 ({expected_tokens})")
    return {
        "source_hash": source_hash,
        "vocabulary": vocabulary,
        "vocabulary_sha256": digest,
        "train": train,
        "validation": validation,
    }


class NaturalStream:
    """Random contiguous training windows from a seeded generator."""

    def __init__(self, stream: torch.Tensor, *, sequence_length: int, seed: int) -> None:
        if stream.numel() <= sequence_length + 1:
            raise ValueError("training stream is shorter than one window")
        self.stream = stream
        self.sequence_length = int(sequence_length)
        self.generator = torch.Generator().manual_seed(int(seed))

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = self.stream.numel() - self.sequence_length - 1
        starts = torch.randint(maximum + 1, (int(size),), generator=self.generator)
        span = torch.arange(self.sequence_length + 1)
        windows = self.stream[starts[:, None] + span[None, :]]
        return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def validation_windows(stream: torch.Tensor, sequence_length: int) -> torch.Tensor:
    count = (stream.numel() - 1) // sequence_length
    starts = torch.arange(count) * sequence_length
    span = torch.arange(sequence_length + 1)
    return stream[starts[:, None] + span[None, :]]


def unigram_nll(train: torch.Tensor, windows: torch.Tensor, vocab_size: int) -> float:
    counts = torch.bincount(train, minlength=vocab_size).double() + 1.0
    log_probabilities = (counts / counts.sum()).log()
    return float((-log_probabilities[windows[:, 1:]]).mean().item())


def content_range(vocab_size: int) -> tuple[int, int]:
    """Token ids usable as synthetic symbols: every non-special id."""
    return len(SPECIAL_TOKENS), int(vocab_size)
