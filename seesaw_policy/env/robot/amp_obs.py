"""AMP discriminator observation from mjlab Go2 state.

Matches the field order of Go2Robot.get_amp_observations. This does not
implement AMP replay, reference initialization, or discriminator training.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

FOOT_BODY_NAMES = tuple(f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR"))


def amp_observation(env, *, include_base_lin_vel: bool = True) -> torch.Tensor:
    entity = env.scene["robot"]
    state = entity.data
    foot_ids, found = entity.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    if tuple(found) != FOOT_BODY_NAMES:
        raise ValueError(f"AMP foot body order differs: {found}")
    rel_world = state.body_link_pos_w[:, foot_ids] - state.root_link_pos_w[:, None]
    base_quat = state.root_link_quat_w[:, None].expand(-1, 4, -1)
    feet_base = quat_apply_inverse(base_quat, rel_world).reshape(env.num_envs, 12)
    parts = [state.joint_pos, feet_base]
    if include_base_lin_vel:
        parts.append(state.root_link_lin_vel_b)
    parts.extend(
        [state.root_link_ang_vel_b, state.joint_vel, state.root_link_pos_w[:, 2:3]]
    )
    obs = torch.cat(parts, dim=-1)
    expected = 43 if include_base_lin_vel else 40
    if obs.shape != (env.num_envs, expected):
        raise AssertionError(f"AMP observation shape changed: {tuple(obs.shape)}")
    return obs
