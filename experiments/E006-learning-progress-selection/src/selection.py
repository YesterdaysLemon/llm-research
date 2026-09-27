"""Learning-progress scores and selection rules for E006.

The score follows Eq. 2 of arXiv:2609.30063, applied to natural-text windows:

    s_i = < grad L(y_i; theta_t), P_t * (theta_anchor - theta_t) >

where P_t = lr / (sqrt(v_hat_t) + eps) is AdamW's diagonal step operator and
theta_anchor is the latest stored snapshot at or before step floor(t / 2).
To first order, s_i = L(y_i; theta_anchor) - L(y_i; theta_t): positive when the
learner has improved on y_i over the lookback window.

Because the direction is shared by all candidates, one forward-mode pass over
the pool yields every s_i.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, grad, jvp, vmap

ARMS = ("uniform", "loss", "gradnorm", "cosine", "signed", "absolute")

# Which per-step statistic each arm needs (the method's own cost).
ARM_REQUIREMENT = {
    "uniform": "none",
    "loss": "forward",
    "gradnorm": "per_sample",
    "cosine": "per_sample",
    "signed": "jvp",
    "absolute": "jvp",
}


def parameter_dict(model: nn.Module) -> dict[str, Tensor]:
    """Named trainable parameters; tied weights appear once."""
    return dict(model.named_parameters())


def detached(params: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach() for name, value in params.items()}


def sequence_losses(
    model: nn.Module, params: dict[str, Tensor], inputs: Tensor, targets: Tensor
) -> Tensor:
    """Mean next-token cross-entropy of each sequence, in at least float32."""
    logits, _ = functional_call(model, params, (inputs,))
    if logits.dtype not in (torch.float32, torch.float64):
        logits = logits.float()
    losses = F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    )
    return losses.view_as(targets).mean(dim=1)


class SnapshotBank:
    """Parameter snapshots every `interval` steps, kept on the CPU.

    `step` counts completed optimizer updates, so the snapshot at step 0 is the
    initialization. The anchor for step t is the snapshot at
    interval * floor(floor(t / 2) / interval).
    """

    def __init__(self, interval: int) -> None:
        if interval < 1:
            raise ValueError("snapshot interval must be positive")
        self.interval = int(interval)
        self.snapshots: dict[int, dict[str, Tensor]] = {}
        self._device_cache: tuple[int, dict[str, Tensor]] | None = None

    def anchor_step(self, step: int) -> int:
        return (int(step) // 2 // self.interval) * self.interval

    def store(self, step: int, params: dict[str, Tensor]) -> None:
        if step % self.interval == 0:
            self.snapshots[int(step)] = {
                name: value.detach().to("cpu", copy=True)
                for name, value in params.items()
            }

    def prune(self, step: int) -> None:
        keep_from = self.anchor_step(step)
        for stored in [key for key in self.snapshots if key < keep_from]:
            del self.snapshots[stored]

    def anchor(self, step: int, device: torch.device) -> tuple[int, dict[str, Tensor]]:
        target = self.anchor_step(step)
        if self._device_cache is None or self._device_cache[0] != target:
            if target not in self.snapshots:
                raise KeyError(f"no snapshot stored for anchor step {target}")
            self._device_cache = (
                target,
                {name: value.to(device) for name, value in self.snapshots[target].items()},
            )
        return self._device_cache


def adam_step_operator(
    optimizer: torch.optim.Optimizer, params: dict[str, Tensor]
) -> dict[str, Tensor]:
    """AdamW's diagonal step operator lr / (sqrt(v_hat) + eps) for each parameter."""
    if len(optimizer.param_groups) != 1:
        raise ValueError("expected a single parameter group")
    group = optimizer.param_groups[0]
    lr = float(group["lr"])
    beta2 = float(group["betas"][1])
    eps = float(group["eps"])
    operator: dict[str, Tensor] = {}
    for name, parameter in params.items():
        state = optimizer.state[parameter]
        step = float(state["step"])
        v_hat = state["exp_avg_sq"] / (1.0 - beta2**step)
        operator[name] = lr / (v_hat.sqrt() + eps)
    return operator


@dataclass
class Direction:
    tangent: dict[str, Tensor]  # P * (theta_anchor - theta_t)
    operator: dict[str, Tensor]  # P
    anchor_step: int
    preconditioned_norm: float  # sqrt(sum P * delta^2) = ||P^(1/2) delta||


def learning_direction(
    params: dict[str, Tensor],
    anchor: dict[str, Tensor],
    operator: dict[str, Tensor],
    anchor_step: int,
) -> Direction:
    tangent: dict[str, Tensor] = {}
    squared = torch.zeros((), dtype=torch.float64)
    for name, value in params.items():
        delta = anchor[name] - value.detach()
        tangent[name] = operator[name] * delta
        squared = squared + (operator[name] * delta.square()).sum().double().cpu()
    return Direction(tangent, operator, int(anchor_step), float(squared.sqrt().item()))


def jvp_scores(
    model: nn.Module,
    params: dict[str, Tensor],
    tangent: dict[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-sequence losses and s_i = <grad L_i, tangent> in one forward-mode pass."""
    primals = detached(params)

    def losses(values: dict[str, Tensor]) -> Tensor:
        return sequence_losses(model, values, inputs, targets)

    value, derivative = jvp(losses, (primals,), (tangent,))
    return value.detach(), derivative.detach()


def per_sample_statistics(
    model: nn.Module,
    params: dict[str, Tensor],
    direction: Direction,
    inputs: Tensor,
    targets: Tensor,
    *,
    chunk: int,
) -> dict[str, Tensor]:
    """Per-sequence loss, ||P^(1/2) g_i||, and <g_i, tangent> from per-sample gradients."""
    primals = detached(params)

    def single(values: dict[str, Tensor], x: Tensor, y: Tensor) -> Tensor:
        return sequence_losses(model, values, x.unsqueeze(0), y.unsqueeze(0))[0]

    batched = vmap(grad(single), in_dims=(None, 0, 0))
    norms: list[Tensor] = []
    dots: list[Tensor] = []
    for start in range(0, inputs.shape[0], int(chunk)):
        stop = start + int(chunk)
        gradients = batched(primals, inputs[start:stop], targets[start:stop])
        squared = sum(
            (direction.operator[name] * gradient.square()).flatten(1).sum(1)
            for name, gradient in gradients.items()
        )
        dot = sum(
            (gradient * direction.tangent[name]).flatten(1).sum(1)
            for name, gradient in gradients.items()
        )
        norms.append(squared.sqrt())
        dots.append(dot)
    with torch.no_grad():
        losses = sequence_losses(model, primals, inputs, targets)
    return {
        "loss": losses.detach(),
        "gradnorm": torch.cat(norms).detach(),
        "dot": torch.cat(dots).detach(),
    }


def cosine_from(dot: Tensor, gradnorm: Tensor, direction_norm: float) -> Tensor:
    """|cos| between P^(1/2) g_i and P^(1/2) delta."""
    denominator = (gradnorm * float(direction_norm)).clamp_min(1e-30)
    return dot.abs() / denominator


def top_indices(scores: Tensor, count: int) -> Tensor:
    """Indices of the `count` largest scores, ties broken by pool order."""
    order = torch.sort(scores.detach().cpu(), descending=True, stable=True).indices
    return order[: int(count)].sort().values


def arm_scores(arm: str, statistics: dict[str, Tensor], direction_norm: float) -> Tensor:
    if arm == "loss":
        return statistics["loss"]
    if arm == "gradnorm":
        return statistics["gradnorm"]
    if arm == "cosine":
        return cosine_from(statistics["dot"], statistics["gradnorm"], direction_norm)
    if arm == "signed":
        return statistics["jvp"]
    if arm == "absolute":
        return statistics["jvp"].abs()
    raise ValueError(f"arm {arm!r} has no score")


def spearman(left: Tensor, right: Tensor) -> float:
    """Spearman rank correlation without tie correction (scores are continuous)."""

    def ranks(values: Tensor) -> Tensor:
        order = torch.argsort(values.detach().double().cpu(), stable=True)
        result = torch.empty_like(order, dtype=torch.float64)
        result[order] = torch.arange(values.numel(), dtype=torch.float64)
        return result

    a, b = ranks(left), ranks(right)
    a, b = a - a.mean(), b - b.mean()
    denominator = (a.square().sum() * b.square().sum()).sqrt()
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")
