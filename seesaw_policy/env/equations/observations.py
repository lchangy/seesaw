"""privileged Teacher Actor, Critic, and AMP observation packing.

Field order and noise placement mirror the captured Gym task.  The functions
depend only on PyTorch tensors so the Isaac Lab task can build simulator state
separately and pass it through this checked policy boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .config import ConfigView


def _matrix(name: str, value: torch.Tensor, rows: int, width: int) -> None:
    if not torch.is_tensor(value) or tuple(value.shape) != (rows, width):
        shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
        raise ValueError(f"{name} must have shape {(rows, width)}, got {shape}")


def quat_rotate_inverse_wxyz(quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse of scalar-first Isaac Lab quaternions."""

    if quat.shape[-1] != 4 or vectors.shape[-1] != 3:
        raise ValueError("quaternion/vector widths must be 4/3")
    vector_part = quat[..., 1:]
    scalar_part = quat[..., :1]
    first_cross = torch.cross(vector_part, vectors, dim=-1)
    return (
        vectors
        - 2.0 * scalar_part * first_cross
        + 2.0 * torch.cross(vector_part, first_cross, dim=-1)
    )


def build_actor_observation(
    *,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    commands: torch.Tensor,
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    recipe: ConfigView,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build the 45-D proprioceptive observation used by the privileged Actor."""

    rows = actions.shape[0]
    for name, value in (
        ("base_ang_vel", base_ang_vel),
        ("projected_gravity", projected_gravity),
        ("commands", commands),
    ):
        _matrix(name, value, rows, 3)
    for name, value in (
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
        ("actions", actions),
    ):
        _matrix(name, value, rows, 12)
    if default_joint_pos.device != actions.device or default_joint_pos.shape not in (
        (12,),
        (rows, 12),
    ):
        raise ValueError("default_joint_pos must have shape [12] or [num_envs, 12]")

    env = recipe.env
    scales = env["normalization"]["obs_scales"]
    command_scale = commands.new_tensor(
        [float(scales["lin_vel"]), float(scales["lin_vel"]), float(scales["ang_vel"])]
    )
    policy = torch.cat(
        (
            base_ang_vel * float(scales["ang_vel"]),
            projected_gravity,
            commands * command_scale,
            (joint_pos - default_joint_pos) * float(scales["dof_pos"]),
            joint_vel * float(scales["dof_vel"]),
            actions,
        ),
        dim=-1,
    )

    noise_cfg = env["noise"]
    if noise_cfg.get("add_noise", False):
        level = float(noise_cfg["noise_level"])
        noise_scales = noise_cfg["noise_scales"]
        amplitude = torch.zeros_like(policy)
        amplitude[:, 0:3] = (
            float(noise_scales["ang_vel"]) * level * float(scales["ang_vel"])
        )
        amplitude[:, 3:6] = float(noise_scales["gravity"]) * level
        amplitude[:, 9:21] = (
            float(noise_scales["dof_pos"]) * level * float(scales["dof_pos"])
        )
        amplitude[:, 21:33] = (
            float(noise_scales["dof_vel"]) * level * float(scales["dof_vel"])
        )
        perturbation = torch.rand(
            policy.shape,
            dtype=policy.dtype,
            device=policy.device,
            generator=generator,
        )
        policy = policy + (2.0 * perturbation - 1.0) * amplitude

    clip = float(env["normalization"]["clip_observations"])
    policy = policy.clamp(min=-clip, max=clip)
    if policy.shape[1] != int(env["env"]["num_observations"]):
        raise ValueError(f"Actor observation width is {policy.shape[1]}, expected 45")
    return policy


def build_amp_observation(
    *,
    joint_pos: torch.Tensor,
    feet_pos_world: torch.Tensor,
    root_pos_world: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
    recipe: ConfigView,
) -> torch.Tensor:
    """Build AMP expert features in the captured order and root frame."""

    rows = joint_pos.shape[0]
    _matrix("joint_pos", joint_pos, rows, 12)
    _matrix("base_lin_vel", base_lin_vel, rows, 3)
    _matrix("base_ang_vel", base_ang_vel, rows, 3)
    _matrix("joint_vel", joint_vel, rows, 12)
    _matrix("root_pos_world", root_pos_world, rows, 3)
    _matrix("root_quat_wxyz", root_quat_wxyz, rows, 4)
    if tuple(feet_pos_world.shape) != (rows, 4, 3):
        raise ValueError(f"feet_pos_world must have shape {(rows, 4, 3)}")

    relative_feet = feet_pos_world - root_pos_world[:, None, :]
    feet_in_base = quat_rotate_inverse_wxyz(root_quat_wxyz[:, None, :], relative_feet)
    fields = [joint_pos, feet_in_base.flatten(1)]
    if bool(recipe.env["env"].get("amp_include_base_lin_vel", True)):
        fields.append(base_lin_vel)
    fields.extend((base_ang_vel, joint_vel, root_pos_world[:, 2:3]))
    amp = torch.cat(fields, dim=-1)
    expected = recipe.train["runner"]["amp_num_frames"]
    if int(expected) != 2:
        raise ValueError(f"Captured privileged AMP expects two frames, got {expected}")
    return amp


def build_critic_fields(
    state: Mapping[str, Any], recipe: ConfigView
) -> dict[str, torch.Tensor]:
    """Encode raw DirectRLEnv state into the captured shared-privileged fields.

    The task supplies measured simulator tensors by name. Missing fields fail
    immediately, so an unavailable sensor cannot silently become a zero input.
    """

    required = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "joint_pos",
        "default_joint_pos",
        "joint_vel",
        "last_joint_vel",
        "actions",
        "torques",
        "torque_limits",
        "root_pos_world",
        "root_quat_wxyz",
        "feet_pos_world",
        "feet_vel_world",
        "feet_contact_force_world",
        "feet_terrain_heights",
        "feet_roughness",
        "body_contact_force_world",
        "terrain_height",
        "motor_strength",
        "pd_stiffness",
        "pd_damping",
        "motor_zero_offsets",
        "action_delay",
        "body_masses",
        "restitution",
        "payload_mass",
        "base_com_offset",
        "friction",
        "recovery_state",
        "previous_actions",
        "velocity_impulse_world",
        "external_force_world",
        "external_torque_world",
        "spatial_rays",
        "executed_target",
        "contact_duration",
        "air_duration",
        "episode_length",
        "dt",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"Raw Critic state is missing fields: {missing}")

    actions = state["actions"]
    if not torch.is_tensor(actions) or actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError("actions must have shape [num_envs, 12]")
    rows, device = actions.shape[0], actions.device

    def tensor(name: str) -> torch.Tensor:
        value = state[name]
        if not torch.is_tensor(value) or value.device != device:
            raise ValueError(f"{name} must be a tensor on {device}")
        return value

    def matrix(name: str, width: int) -> torch.Tensor:
        value = tensor(name)
        _matrix(name, value, rows, width)
        return value

    scales = recipe.env["normalization"]["obs_scales"]
    command_scale = actions.new_tensor(
        [float(scales["lin_vel"]), float(scales["lin_vel"]), float(scales["ang_vel"])]
    )
    root_pos = matrix("root_pos_world", 3)
    root_quat = matrix("root_quat_wxyz", 4)
    feet_pos = tensor("feet_pos_world")
    feet_vel = tensor("feet_vel_world")
    feet_force = tensor("feet_contact_force_world")
    for name, value in (
        ("feet_pos_world", feet_pos),
        ("feet_vel_world", feet_vel),
        ("feet_contact_force_world", feet_force),
    ):
        if tuple(value.shape) != (rows, 4, 3):
            raise ValueError(f"{name} must have shape {(rows, 4, 3)}")
    feet_rotation = root_quat[:, None, :]
    feet_position_base = quat_rotate_inverse_wxyz(
        feet_rotation, feet_pos - root_pos[:, None, :]
    ).flatten(1)
    feet_velocity_base = quat_rotate_inverse_wxyz(feet_rotation, feet_vel).flatten(1)
    feet_force_base = quat_rotate_inverse_wxyz(feet_rotation, feet_force).flatten(1)

    feet_heights = tensor("feet_terrain_heights")
    feet_roughness = tensor("feet_roughness")
    if tuple(feet_heights.shape) != (rows, 4, 9) or tuple(feet_roughness.shape) != (
        rows,
        4,
    ):
        raise ValueError(
            "feet_terrain_heights must be [num_envs, 4, 9] and roughness [num_envs, 4]"
        )
    episode_length = tensor("episode_length").reshape(rows)
    feet_state_valid = (episode_length > 0).to(actions.dtype).unsqueeze(1)
    height_scale = float(scales["height_measurements"])
    body_contact_force = tensor("body_contact_force_world")
    contact_width = int(
        recipe.train["policy"]["critic_privileged_layout"]["body_contact"]
    )
    if tuple(body_contact_force.shape) != (rows, contact_width, 3):
        raise ValueError(
            "body_contact_force_world must have shape "
            f"{(rows, contact_width, 3)}, got {tuple(body_contact_force.shape)}"
        )
    contact_threshold = float(
        recipe.env["env"].get("privileged_body_contact_threshold", 1.0)
    )
    body_contact = (
        torch.linalg.vector_norm(body_contact_force, dim=-1) > contact_threshold
    ).to(actions.dtype)

    torque_limits = tensor("torque_limits")
    torques = matrix("torques", 12)
    if torque_limits.shape not in ((12,), (rows, 12)):
        raise ValueError("torque_limits must have shape [12] or [num_envs, 12]")
    default_joint_pos = tensor("default_joint_pos")
    if default_joint_pos.shape not in ((12,), (rows, 12)):
        raise ValueError("default_joint_pos must have shape [12] or [num_envs, 12]")
    dt = float(state["dt"])
    if dt <= 0:
        raise ValueError("dt must be positive")

    impulse_world = matrix("velocity_impulse_world", 6)
    impulse_linear = quat_rotate_inverse_wxyz(root_quat, impulse_world[:, :3])
    impulse_angular = quat_rotate_inverse_wxyz(root_quat, impulse_world[:, 3:])
    force_base = quat_rotate_inverse_wxyz(root_quat, matrix("external_force_world", 3))
    torque_base = quat_rotate_inverse_wxyz(
        root_quat, matrix("external_torque_world", 3)
    )
    delay = matrix("action_delay", 2)

    fifo_length = int(recipe.env["geometry"]["action_fifo_length"])
    if "action_fifo" in state:
        action_fifo = tensor("action_fifo")
        if tuple(action_fifo.shape) != (rows, fifo_length * 12):
            raise ValueError(
                f"action_fifo must have shape {(rows, fifo_length * 12)}, "
                f"got {tuple(action_fifo.shape)}"
            )
    else:
        action_fifo = actions.new_zeros(rows, fifo_length * 12)
        action_fifo[:, -12:] = actions

    contact_duration = tensor("contact_duration")
    air_duration = tensor("air_duration")
    if tuple(contact_duration.shape) != (rows, 4) or tuple(air_duration.shape) != (
        rows,
        4,
    ):
        raise ValueError(
            "contact_duration and air_duration must have shape [num_envs, 4]"
        )

    fields = {
        "base_linear_velocity": matrix("base_lin_vel", 3) * float(scales["lin_vel"]),
        "base_angular_velocity": matrix("base_ang_vel", 3) * float(scales["ang_vel"]),
        "gravity": matrix("projected_gravity", 3),
        "command": matrix("commands", 3) * command_scale,
        "joint_position": (matrix("joint_pos", 12) - default_joint_pos)
        * float(scales["dof_pos"]),
        "joint_velocity": matrix("joint_vel", 12) * float(scales["dof_vel"]),
        "action": actions,
        "joint_torque": torques / torque_limits,
        "negative_joint_acceleration": (
            matrix("last_joint_vel", 12) - matrix("joint_vel", 12)
        )
        / dt
        * 1e-4,
        "terrain_height": matrix(
            "terrain_height",
            int(recipe.train["policy"]["critic_privileged_layout"]["terrain_height"]),
        ),
        "body_contact": body_contact,
        "friction": matrix("friction", 1),
        "motor_strength": matrix("motor_strength", 12),
        "pd_stiffness": matrix("pd_stiffness", 12) / 20.0,
        "pd_damping": matrix("pd_damping", 12) / 0.5,
        "motor_zero_offset": matrix("motor_zero_offsets", 12) * 10.0,
        "action_delay": delay,
        "body_masses": matrix(
            "body_masses",
            int(recipe.train["policy"]["critic_privileged_layout"]["body_masses"]),
        )
        * 0.1,
        "restitution": matrix("restitution", 1),
        "actual_payload": matrix("payload_mass", 1) * 0.1,
        "actual_base_com": matrix("base_com_offset", 3),
        "feet_velocity": feet_velocity_base * float(scales["lin_vel"]),
        "feet_contact_force": feet_force_base * 1e-3,
        "feet_position": feet_position_base,
        "feet_local_height": (feet_heights - feet_pos[:, :, 2:3]).flatten(1)
        * height_scale,
        "feet_roughness": feet_roughness * height_scale,
        "feet_state_valid": feet_state_valid,
        "recovery_state": matrix("recovery_state", 9),
        "previous_action": matrix("previous_actions", 12),
        "velocity_impulse": torch.cat((impulse_linear, impulse_angular), dim=1),
        "external_force": force_base / 50.0,
        "external_torque": torque_base / 10.0,
        "spatial_rays": matrix("spatial_rays", 260),
        "executed_target": (matrix("executed_target", 12) - default_joint_pos)
        * float(scales["dof_pos"]),
        "action_fifo": action_fifo,
        "contact_duration": contact_duration.clamp(max=3.0),
        "air_duration": air_duration.clamp(max=3.0),
    }

    valid = feet_state_valid
    for name in (
        "body_contact",
        "feet_velocity",
        "feet_contact_force",
        "feet_position",
        "feet_local_height",
        "feet_roughness",
    ):
        fields[name] = fields[name] * valid
    return fields


def build_critic_observation(
    fields: Mapping[str, torch.Tensor], recipe: ConfigView
) -> torch.Tensor:
    """Pack the 1512-D privileged vector in its frozen manifest field order."""

    layout = recipe.train["policy"]["critic_privileged_layout"]
    missing = set(layout) - fields.keys()
    extra = fields.keys() - set(layout)
    if missing or extra:
        raise ValueError(
            f"Critic field set differs from recipe: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    rows: int | None = None
    pieces: list[torch.Tensor] = []
    device = None
    for name, width in layout.items():
        value = fields[name]
        if not torch.is_tensor(value) or value.ndim != 2:
            shape = (
                tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            )
            raise ValueError(
                f"Critic field {name!r} must be a rank-2 tensor, got {shape}"
            )
        if value.shape[1] != int(width):
            raise ValueError(
                f"Critic field {name!r} has width {value.shape[1]}, expected {int(width)}"
            )
        if rows is None:
            rows, device = value.shape[0], value.device
        elif value.shape[0] != rows or value.device != device:
            raise ValueError("All Critic fields must share batch size and device")
        pieces.append(value)
    critic = torch.cat(pieces, dim=-1)
    expected = int(recipe.env["env"]["num_privileged_obs"])
    if critic.shape[1] != expected:
        raise ValueError(
            f"Critic observation width is {critic.shape[1]}, expected {expected}"
        )
    clip = float(recipe.env["normalization"]["clip_observations"])
    return critic.clamp(min=-clip, max=clip)


def observation_dict(
    policy: torch.Tensor, critic: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return the DirectRLEnv observation keys required by the RSL adapter."""

    if policy.shape[0] != critic.shape[0] or policy.device != critic.device:
        raise ValueError(
            "policy and critic observations must share batch size and device"
        )
    return {"policy": policy, "critic": critic}
