from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, values: Tensor) -> Tensor:
        batch, length, width = values.shape
        qkv = self.qkv(values).view(
            batch, length, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        merged = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(merged)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.dropout(self.attention(self.attention_norm(values)))
        values = values + self.dropout(self.mlp(self.mlp_norm(values)))
        return values


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: Tensor, *, return_hidden: bool = False
    ) -> tuple[Tensor, list[Tensor]]:
        if tokens.shape[1] > self.sequence_length:
            raise ValueError("input exceeds configured sequence length")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        values = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden: list[Tensor] = []
        for block in self.blocks:
            values = block(values)
            if return_hidden:
                hidden.append(values)
        logits = self.lm_head(self.final_norm(values))
        return logits, hidden


def model_from_config(
    spec: dict[str, Any], *, vocab_size: int, sequence_length: int
) -> TinyCausalLM:
    return TinyCausalLM(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        d_model=int(spec["d_model"]),
        n_heads=int(spec["n_heads"]),
        n_layers=int(spec["n_layers"]),
        d_ff=int(spec["d_ff"]),
        dropout=float(spec["dropout"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def centered_relation(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    values = F.normalize(values.float(), dim=-1, eps=epsilon)
    gram = values @ values.transpose(0, 1)
    gram = (
        gram
        - gram.mean(dim=0, keepdim=True)
        - gram.mean(dim=1, keepdim=True)
        + gram.mean()
    )
    return gram / gram.norm().clamp_min(epsilon)


def target_relation(labels: Tensor, epsilon: float = 1e-8) -> Tensor:
    gram = labels[:, None].eq(labels[None, :]).float()
    gram = (
        gram
        - gram.mean(dim=0, keepdim=True)
        - gram.mean(dim=1, keepdim=True)
        + gram.mean()
    )
    return gram / gram.norm().clamp_min(epsilon)


def relation_distance(left: Tensor, right: Tensor) -> Tensor:
    return (left - right).square().sum()


def learning_rate_multiplier(
    step: int, *, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine

