"""Quarantine invalid MJLab terminal worlds without changing healthy rewards.

The manager computes rewards after termination checks but before auto-reset.
Never evaluate a reward equation on invalid physics, and never hide an error
in a healthy world's reward equation with nan_to_num.
"""

from collections.abc import Mapping

import torch

from seesaw_policy.env.equations.rewards import RewardBatch


def finite_worlds(*values: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(values[0].shape[0], dtype=torch.bool, device=values[0].device)
    for value in values:
        mask &= torch.isfinite(value).flatten(1).all(1)
    return mask


def mark_invalid_reward_reset(env, invalid: torch.Tensor) -> None:
    """Turn a late reward failure into a real non-timeout manager reset."""
    env.termination_manager.terminated[invalid] = True
    env.termination_manager.get_term("geometry_task")[invalid] = True
    env.reset_buf[invalid] = True
    env.reset_time_outs[invalid] = False


def terminal_safe_reward(model, state, invalid, terminated):
    """Evaluate only healthy worlds; invalid terminal transitions earn zero.

    Invalid worlds are genuine terminations, never timeouts. Their previous
    finite observation/action can enter PPO, with zero environment reward and
    no value bootstrap. The manager resets physics before producing next obs.
    """
    if not bool(invalid.any()):
        return model.compute(state, quarantine_nonfinite=True)
    if bool((invalid & ~terminated).any()):
        raise RuntimeError("Invalid physics must terminate before reward evaluation")
    rows = invalid.numel()
    good = ~invalid

    def select(value):
        if isinstance(value, Mapping):
            return {key: select(item) for key, item in value.items()}
        if torch.is_tensor(value) and value.ndim and value.shape[0] == rows:
            return value[good]
        return value

    def expand(value):
        result = value.new_zeros(rows)
        result[good] = value
        return result

    if not bool(good.any()):
        zero = state["actions"].new_zeros(rows)
        names = (
            set(model.scales)
            | set(model.recovery_scales)
            | set(state.get("parkour_terms", {}))
        )
        return RewardBatch(
            zero,
            zero.clone(),
            zero.clone(),
            {name: zero.clone() for name in names},
            {name: zero.clone() for name in names},
        )
    batch = model.compute(select(state), quarantine_nonfinite=True)
    invalid_rows = (
        expand(batch.invalid_rows) if batch.invalid_rows is not None else None
    )
    return RewardBatch(
        expand(batch.total),
        expand(batch.task),
        expand(batch.auxiliary),
        {key: expand(value) for key, value in batch.terms.items()},
        {key: expand(value) for key, value in batch.raw_terms.items()},
        invalid_rows.bool() if invalid_rows is not None else None,
    )
