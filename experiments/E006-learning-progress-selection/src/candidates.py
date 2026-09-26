"""Natural-text data and the per-step candidate pool for E006.

The tokenizer, story split, and packing are imported unchanged from the
completed E004 experiment so that E006 reproduces E004's vocabulary exactly.
"""

from __future__ import annotations

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
    import hashlib

    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def prepare_natural_data(dataset: dict[str, Any]) -> dict[str, Any]:
    """Load the pinned TinyStories artifact and verify it against the config."""
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
        raise RuntimeError(
            f"vocabulary SHA-256 {digest} does not reproduce E004 ({expected})"
        )
    train = pack_stories(train_stories, vocabulary)
    validation = pack_stories(validation_stories, vocabulary)
    for key, stream in (("train", train), ("validation", validation)):
        expected_tokens = dataset.get(f"expected_{key}_tokens")
        if expected_tokens is not None and stream.numel() != int(expected_tokens):
            raise RuntimeError(
                f"{key} token count {stream.numel()} does not reproduce E004 ({expected_tokens})"
            )
    return {
        "source_hash": source_hash,
        "vocabulary": vocabulary,
        "vocabulary_sha256": digest,
        "train": train,
        "validation": validation,
    }


class CandidatePool:
    """Deterministic stream of candidate windows, identical across selection arms.

    Every call to `draw` consumes the same random numbers whatever the caller
    later selects, so arms that share a seed see the same pool at every step.
    Window offsets use one generator and noise decisions another, so the clean
    windows of a noisy pool are exactly the windows of the clean pool.
    """

    def __init__(
        self,
        stream: torch.Tensor,
        *,
        sequence_length: int,
        vocab_size: int,
        seed: int,
        noise_fraction: float = 0.0,
        noise_seed: int | None = None,
    ) -> None:
        if stream.numel() <= sequence_length + 1:
            raise ValueError("training stream is shorter than one window")
        if not 0.0 <= noise_fraction < 1.0:
            raise ValueError("noise_fraction must be in [0, 1)")
        self.stream = stream
        self.sequence_length = int(sequence_length)
        self.vocab_size = int(vocab_size)
        self.noise_fraction = float(noise_fraction)
        self.offsets = torch.Generator().manual_seed(int(seed))
        self.noise = torch.Generator().manual_seed(
            int(seed if noise_seed is None else noise_seed)
        )
        self.first_content_id = len(SPECIAL_TOKENS)

    def draw(self, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return inputs [size, T], targets [size, T], and a boolean noise mask."""
        maximum = self.stream.numel() - self.sequence_length - 1
        starts = torch.randint(maximum + 1, (int(size),), generator=self.offsets)
        span = torch.arange(self.sequence_length + 1)
        windows = self.stream[starts[:, None] + span[None, :]].clone()
        coins = torch.rand(int(size), generator=self.noise)
        replacement = torch.randint(
            self.first_content_id,
            self.vocab_size,
            windows.shape,
            generator=self.noise,
        )
        noise_mask = coins < self.noise_fraction
        windows[noise_mask] = replacement[noise_mask]
        return windows[:, :-1].contiguous(), windows[:, 1:].contiguous(), noise_mask


def validation_windows(stream: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """All non-overlapping (T + 1)-token windows of the validation stream."""
    count = (stream.numel() - 1) // sequence_length
    starts = torch.arange(count) * sequence_length
    span = torch.arange(sequence_length + 1)
    return stream[starts[:, None] + span[None, :]]


def unigram_nll(train: torch.Tensor, windows: torch.Tensor, vocab_size: int) -> float:
    """Add-one-smoothed training unigram NLL on the validation targets."""
    counts = torch.bincount(train, minlength=vocab_size).double() + 1.0
    log_probabilities = (counts / counts.sum()).log()
    return float((-log_probabilities[windows[:, 1:]]).mean().item())
