"""Identify actor input rows that cannot safely enter the policy network."""

import torch


def invalid_policy_rows(
    obs: torch.Tensor, privileged: torch.Tensor, *, max_magnitude: float = 1.0e4
) -> torch.Tensor:
    """Return rows with nonfinite or implausibly large actor/critic features."""
    if obs.ndim != 2 or privileged.ndim != 2 or obs.shape[0] != privileged.shape[0]:
        raise ValueError("Expected matching batched actor and privileged observations")
    actor_ok = torch.isfinite(obs).all(dim=1) & (obs.abs() <= max_magnitude).all(dim=1)
    critic_ok = torch.isfinite(privileged).all(dim=1) & (
        privileged.abs() <= max_magnitude
    ).all(dim=1)
    return ~(actor_ok & critic_ok)
