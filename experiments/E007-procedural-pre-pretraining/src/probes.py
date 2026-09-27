"""In-context probes: does a skill survive into the natural-text model?

Both probes use word tokens the model never saw arranged this way in training,
so any success must be inferred from the prompt itself.

* `copy`: a random sequence of distinct words followed by its repeat. Score:
  greedy accuracy on the second copy from its second position on (the first
  repeated word cannot be anticipated).
* `recall`: key/value word pairs, then a query key. Score: greedy accuracy on
  the queried key's value.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def probe_prompts(params: dict[str, Any]) -> dict[str, list[tuple[list[int], list[int], list[int]]]]:
    """Deterministic probe prompts: (tokens, scored positions, expected next tokens)."""
    generator = torch.Generator().manual_seed(int(params["seed"]))
    low, high = int(params["token_low"]), int(params["token_high"])
    trials = int(params["trials"])
    copy_length = int(params["copy_length"])
    pairs = int(params["recall_pairs"])
    copy: list[tuple[list[int], list[int], list[int]]] = []
    for _ in range(trials):
        words = (low + torch.randperm(high - low, generator=generator)[:copy_length]).tolist()
        tokens = words + words
        positions = [copy_length + index - 1 for index in range(1, copy_length)]
        copy.append((tokens, positions, words[1:]))
    recall: list[tuple[list[int], list[int], list[int]]] = []
    for _ in range(trials):
        drawn = (low + torch.randperm(high - low, generator=generator)[: 2 * pairs]).tolist()
        keys, values = drawn[:pairs], drawn[pairs:]
        query = int(torch.randint(pairs, (), generator=generator))
        tokens = [token for pair in zip(keys, values) for token in pair] + [keys[query]]
        recall.append((tokens, [len(tokens) - 1], [values[query]]))
    return {"copy": copy, "recall": recall}


@torch.no_grad()
def evaluate_probes(
    model: nn.Module,
    prompts: dict[str, list[tuple[list[int], list[int], list[int]]]],
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    scores: dict[str, float] = {}
    for name, rows in prompts.items():
        tokens = torch.tensor([row[0] for row in rows], device=device)
        logits, _ = model(tokens)
        predictions = logits.argmax(dim=-1).cpu()
        correct = 0
        total = 0
        for index, (_, positions, expected) in enumerate(rows):
            for position, answer in zip(positions, expected):
                correct += int(predictions[index, position] == answer)
                total += 1
        scores[name] = correct / total
    model.train()
    return scores
