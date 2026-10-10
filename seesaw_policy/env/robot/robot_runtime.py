"""Robot state, rewards, resets and domain randomization for seesaw.

This module is used only by ``seesaw_policy.env.robot``.  The frozen Gym
recipes and simulator-independent privileged observation/reward equations are
inputs; the live state and event handling are native MJLab/MuJoCo-Warp.
"""

from __future__ import annotations
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from seesaw_policy.env.equations.observations import (
    build_actor_observation,
    build_amp_observation,
    build_critic_fields,
    build_critic_observation,
)
from seesaw_policy.env.equations.rewards import RobotRewardModel, RewardBatch
from .asset import GO2_XML, OFFICIAL_CALF_EFFORT_NM, OFFICIAL_HIP_THIGH_EFFORT_NM
from .foothold import FootholdPlaneResidual
from .numerical_safety import (
    finite_worlds,
    mark_invalid_reward_reset,
    terminal_safe_reward,
)
from .terrain_rays import TerrainRayReader
from .config import TaskConfig
from .robot_randomization import progress as robot_dr_progress

ROOT = Path(__file__).resolve().parents[3]
FOOT_NAMES = tuple((f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")))
BODY_CONTACT_NAMES = (
    "base_link",
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
)


def _quat_to_rpy_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    sin_pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    return torch.stack((roll, pitch, yaw), dim=-1)


def _xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return quat[..., (3, 0, 1, 2)]


def reference_frame_to_reset_state(
    frames: torch.Tensor, origins: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map Gym AMP's local, xyzw/body-velocity frames into MJLab world state."""
    if frames.ndim != 2 or frames.shape[1] < 49 or origins.shape != (len(frames), 3):
        raise ValueError("Invalid AMP reference frame or environment origin shape")
    root = frames.new_zeros((len(frames), 13))
    root[:, :3] = frames[:, :3] + origins
    root[:, 3:7] = _xyzw_to_wxyz(frames[:, 3:7])
    root[:, 7:10] = quat_apply(root[:, 3:7], frames[:, 31:34])
    root[:, 10:13] = quat_apply(root[:, 3:7], frames[:, 34:37])
    return (root, frames[:, 7:19].clone(), frames[:, 37:49].clone())


def _event_env_ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def separate_foot_geom_contact_forces(
    body_forces: torch.Tensor,
    feet_forces: torch.Tensor,
    calf_body_ids: torch.Tensor,
    foot_body_ids: torch.Tensor,
) -> torch.Tensor:
    """Assign MJCF foot-geom forces to foot bodies, not their calf parents.

    Unitree's official MJCF places each named foot collision geom on its calf
    body, while the privileged USD exposes a separate foot rigid body.  The full
    body sensor therefore includes each foot force in the calf's net force.
    Subtract it there and put it on the corresponding empty MJCF foot body so
    body-level rewards and the contact-force limit count each impact once.
    """
    if body_forces.ndim != 3 or body_forces.shape[-1] != 3:
        raise ValueError("body_forces must have shape [num_envs, num_bodies, 3]")
    if feet_forces.shape != (body_forces.shape[0], 4, 3):
        raise ValueError("feet_forces must have shape [num_envs, 4, 3]")
    if calf_body_ids.shape != (4,) or foot_body_ids.shape != (4,):
        raise ValueError(
            "calf_body_ids and foot_body_ids must each contain four body indices"
        )
    result = body_forces.clone()
    result[:, calf_body_ids] = result[:, calf_body_ids] - feet_forces
    result[:, foot_body_ids] = feet_forces
    return result


class RobotRuntime:
    """Robot dynamics, observations and reset state for the seesaw task."""

    def __init__(
        self,
        env,
        recipe: TaskConfig,
        pool: str,
        *,
        seed: int | None = None,
        recovery_pool_size: int | None = None,
        active_mask: torch.Tensor | None = None,
        terrain_type_offset: int = 0,
    ):
        self.env = env
        self.task_config = recipe
        self.stage = 3
        self.pool = pool
        self.device = torch.device(env.device)
        self.num_envs = env.num_envs
        self.active_mask = active_mask
        self.terrain_type_offset = int(terrain_type_offset)
        self.dt = float(env.step_dt)
        self.physics_dt = float(env.physics_dt)
        self.recipe_view = SimpleNamespace(
            env=recipe.env_config,
            train=recipe.train_config,
            control_dt=recipe.control_dt,
            stage=3,
        )
        self.reward_model = RobotRewardModel(self.recipe_view, decomposition_rtol=1e-06)
        self.rays = TerrainRayReader(env, recipe)
        height_schema = recipe.train_config["policy"]["critic_terrain_schema"]
        height_x = tuple(
            (
                i
                for i, x in enumerate(height_schema["height_x"])
                if -0.2 - 1e-06 <= x <= 0.2 + 1e-06
            )
        )
        height_y = tuple(
            (
                i
                for i, y in enumerate(height_schema["height_y"])
                if -0.15 - 1e-06 <= y <= 0.15 + 1e-06
            )
        )
        if not height_x or not height_y:
            raise ValueError(
                "privileged height map does not cover the Gym base-height scan"
            )
        self._base_height_x = torch.tensor(
            height_x, dtype=torch.long, device=self.device
        )
        self._base_height_y = torch.tensor(
            height_y, dtype=torch.long, device=self.device
        )
        foot_axis = torch.tensor(height_schema["foot_axis"], device=self.device)
        foot_x, foot_y = torch.meshgrid(foot_axis, foot_axis, indexing="ij")
        self._stair_face_probe_distance = torch.hypot(foot_x, foot_y)
        self._foothold_plane_residual = FootholdPlaneResidual(
            height_schema["foot_axis"],
            float(recipe.env_config["rewards"].get("foothold_footmap_radius", 0.05)),
            self.device,
        )
        self._step_ray_data: dict[str, Any] | None = None
        self.amp_height_valid = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.base_height_valid = torch.ones_like(self.amp_height_valid)
        env.terrain_data = SimpleNamespace(
            schema=recipe.train_config["policy"]["critic_terrain_schema"], data=None
        )
        self.entity = env.scene["robot"]
        self.feet_ids, feet_found = self.entity.find_bodies(
            FOOT_NAMES, preserve_order=True
        )
        if tuple(feet_found) != FOOT_NAMES:
            raise ValueError(f"MJLab Go2 foot order differs: {feet_found}")
        self.body_contact_ids, body_found = self.entity.find_bodies(
            BODY_CONTACT_NAMES, preserve_order=True
        )
        if tuple(body_found) != BODY_CONTACT_NAMES:
            raise ValueError(f"MJLab body contact order differs: {body_found}")
        self.base_body_id = self.entity.body_names.index("base_link")
        self.body_sensor_names = tuple(env.scene["body_ground_contact"].primary_names)
        self.body_names = tuple(self.entity.body_names)
        if self.body_sensor_names != self.body_names:
            raise ValueError(
                f"privileged body contact sensor must cover every robot body in entity order; sensor={self.body_sensor_names}, entity={self.body_names}"
            )
        self._body_contact_ids = torch.as_tensor(
            self.body_contact_ids, dtype=torch.long, device=self.device
        )
        self._calf_body_ids = torch.as_tensor(
            [self.body_names.index(f"{leg}_calf") for leg in ("FL", "FR", "RL", "RR")],
            dtype=torch.long,
            device=self.device,
        )
        self._foot_body_ids = torch.as_tensor(
            [self.body_names.index(f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")],
            dtype=torch.long,
            device=self.device,
        )
        self.joint_names = tuple(self.entity.joint_names)
        if len(self.joint_names) != 12:
            raise ValueError(f"Expected 12 Go2 joints, got {self.joint_names}")
        self._reference_motion_loader = None
        self.torque_limits = torch.tensor(
            [
                OFFICIAL_HIP_THIGH_EFFORT_NM,
                OFFICIAL_HIP_THIGH_EFFORT_NM,
                OFFICIAL_CALF_EFFORT_NM,
            ]
            * 4,
            dtype=torch.float32,
            device=self.device,
        )
        self.joint_velocity_limits = torch.full((12,), 30.0, device=self.device)
        n = self.num_envs
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=self.device)
        self.training_iteration = int(
            self.task_config.env_config.get("domain_rand", {})
            .get("robot_dr_ramp", {})
            .get("startup_iteration", 0)
        )
        self.recovery_reset_type = torch.zeros(n, dtype=torch.long, device=self.device)
        self.recovery_timer = z(n)
        self.recovery_success_hold = z(n)
        self.recovery_start_height = z(n)
        self.recovery_start_tilt = z(n)
        self.recovery_prev_height = z(n)
        self.recovery_prev_tilt = z(n)
        self.recovery_origin_xy = z(n, 2)
        self.recovery_height_progress = z(n)
        self.recovery_flip_progress = z(n)
        self._recovery_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._recovery_failure = torch.zeros_like(self._recovery_success)
        self.recovery_reward_active = torch.zeros_like(self._recovery_success)
        self.last_turn_over_reward_mask = torch.zeros_like(self._recovery_success)
        self.recovery_target_command = z(n, 3)
        self.previous_joint_vel = z(n, 12)
        self.previous_base_lin_vel_b = z(n, 3)
        self.previous_base_ang_vel_b = z(n, 3)
        self.previous_root_lin_vel_w = z(n, 3)
        self.previous_root_ang_vel_w = z(n, 3)
        self.contact_duration = z(n, 4)
        self.air_duration = z(n, 4)
        self.airtime_reward = z(n)
        self.contact_ema = z(n, 4)
        self._history_initialized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.route_max_x = z(n)
        self.route_progress_check_x = z(n)
        self.route_no_progress_s = z(n)
        self.route_progress_event = z(n)
        self.route_stalled = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.route_backtracked = torch.zeros_like(self.route_stalled)
        self.route_heading_bad = torch.zeros_like(self.route_stalled)
        self.route_finished = torch.zeros_like(self.route_stalled)
        self.route_failed = torch.zeros_like(self.route_stalled)
        self.route_overhead_jump = torch.zeros_like(self.route_stalled)
        self.episode_success = torch.zeros_like(self.route_stalled)
        self.episode_failure = torch.zeros_like(self.route_stalled)
        self.last_task_termination = torch.zeros_like(self.route_stalled)
        self.last_fall_event = torch.zeros_like(self.route_stalled)
        self.last_base_contact_event = torch.zeros_like(self.route_stalled)
        self.last_orientation_event = torch.zeros_like(self.route_stalled)
        self.last_velocity_event = torch.zeros_like(self.route_stalled)
        self.last_velocity_linear_event = torch.zeros_like(self.route_stalled)
        self.last_velocity_angular_event = torch.zeros_like(self.route_stalled)
        self.last_velocity_nonfinite_event = torch.zeros_like(self.route_stalled)
        self.last_recovery_timeout_event = torch.zeros_like(self.route_stalled)
        self.friction = torch.ones(n, 1, device=self.device)
        self.restitution = torch.zeros(n, 1, device=self.device)
        self.payload_mass = z(n, 1)
        self.base_com_offset = z(n, 3)
        self.link_mass_ratio = torch.ones(n, 17, device=self.device)
        self.motor_strength = torch.ones(n, 12, device=self.device)
        self.pd_stiffness = torch.ones(n, 12, device=self.device)
        self.pd_damping = torch.ones(n, 12, device=self.device)
        self.motor_zero_offsets = z(n, 12)
        self.action_delay_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.action_delay_max_steps = torch.zeros_like(self.action_delay_steps)
        self.disturbance_wrench_timer = z(n)
        self.external_wrenches = z(n, 6)
        self.velocity_impulse_world = z(n, 6)
        self._structural_dr_initialized = False
        self._domain_initialized = False
        self._model_baseline: dict[str, torch.Tensor] = {}
        self._robot_geom_mask: torch.Tensor | None = None
        self._robot_body_ids: torch.Tensor | None = None
        self._base_model_body_id: int | None = None
        self._base_model_geom_ids: torch.Tensor | None = None
        self.last_reward_batch = RewardBatch(z(n), z(n), z(n), {}, {})
        self.last_raw_reward_terms: dict[str, torch.Tensor] = {}
        self.last_final_reward_terms: dict[str, torch.Tensor] = {}
        self.last_tracking_reward = z(n)
        self._recovery_bank: dict[str, torch.Tensor] | None = None
        self._terrain_assignment_ready = False
        self._recovery_pool_initialized = False
        self._recovery_pool = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._recovery_curriculum_level = 0
        self._recovery_curriculum_results: list[bool] = []
        self._recovery_curriculum_passes = 0
        self._recovery_curriculum_pending: list[bool] = []
        self._recovery_reset_level = torch.full(
            (n,), -1, dtype=torch.long, device=self.device
        )
        self._recovery_pool_size = recovery_pool_size
        self.route_tracking_mean = z(n)
        self.route_tracking_count = z(n)
        self.push_grace_timer = z(n)
        self.push_next_event = z(n)
        self.push_was_recovery = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._rng = torch.Generator(device=self.device)
        runtime_seed = int(seed if seed is not None else 1201)
        pool_offset = 0
        self._rng.manual_seed(runtime_seed + 3 * 97 + pool_offset)
        self._recovery_rng = torch.Generator(device="cpu")
        self._recovery_rng.manual_seed(runtime_seed + 1907 + 3 * 101 + pool_offset)
        self._current_family_strengths = torch.ones(n, 4, device=self.device)
        self._init_model_baselines()

    @property
    def command(self) -> torch.Tensor:
        return self.env.command_manager.get_command("twist")

    def _init_model_baselines(self) -> None:
        model = self.env.sim.model
        for name in (
            "geom_friction",
            "geom_solref",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "body_mass",
            "body_inertia",
            "body_ipos",
        ):
            value = getattr(model, name, None)
            if torch.is_tensor(value):
                self._model_baseline[name] = value.detach().clone()
        entity = self.entity
        self._robot_body_ids = torch.as_tensor(
            entity.indexing.body_ids, dtype=torch.long, device=self.device
        )
        if self._robot_body_ids.numel() != 17:
            raise ValueError(
                f"Official Go2 model body mapping has {self._robot_body_ids.numel()} bodies"
            )
        xml_model = mujoco.MjModel.from_xml_path(str(GO2_XML))
        self._base_model_body_id = int(
            mujoco.mj_name2id(xml_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        )
        self._base_body_local_id = self.entity.body_names.index("base_link")
        if self._base_model_body_id < 0:
            raise ValueError("Official MJCF is missing base_link")
        body_ids = set((int(x) for x in self._robot_body_ids.tolist()))
        geom_body_ids = self.env.sim.mj_model.geom_bodyid
        geom_ids = [
            i for i, body_id in enumerate(geom_body_ids) if int(body_id) in body_ids
        ]
        self._robot_geom_mask = torch.zeros(
            len(geom_body_ids), dtype=torch.bool, device=self.device
        )
        if geom_ids:
            self._robot_geom_mask[torch.as_tensor(geom_ids, device=self.device)] = True
        if "geom_friction" in self._model_baseline:
            geom_field = self._model_baseline["geom_friction"]
            if geom_field.ndim != 3 or geom_field.shape[0] != self.num_envs:
                raise ValueError(
                    f"MuJoCo-Warp geom_friction is not per-environment: {tuple(geom_field.shape)}"
                )

    def initialize_terrain_assignment(self) -> None:
        if self._terrain_assignment_ready:
            return
        terrain = self.env.scene.terrain
        if terrain is not None and terrain.terrain_origins is not None:
            levels = terrain.terrain_levels
            levels.zero_()
            terrain.env_origins[:] = terrain.terrain_origins[
                terrain.terrain_levels, terrain.terrain_types
            ]
            if self.env.scene.env_origins.data_ptr() != terrain.env_origins.data_ptr():
                self.env.scene.env_origins[:] = terrain.env_origins
        self._terrain_assignment_ready = True

    def _record_recovery_outcomes(self, ids: torch.Tensor) -> None:
        if not ids.numel() or not self._recovery_curriculum_enabled():
            return
        completed = ids[
            (self.env.episode_length_buf[ids] > 0)
            & (self._recovery_reset_level[ids] >= 0)
            & (self._recovery_reset_level[ids] == self._recovery_curriculum_level)
        ]
        if not completed.numel():
            return
        order = torch.argsort(completed)
        completed = completed[order]
        results = [bool(value) for value in self._recovery_success[completed].tolist()]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._recovery_curriculum_pending.extend(results)
        else:
            self._consume_recovery_curriculum_results(results)

    def _consume_recovery_curriculum_results(self, results: list[bool]) -> None:
        if not results:
            return
        window = max(
            int(
                self.task_config.env_config.get("staged_curriculum", {}).get(
                    "recovery_reset_window", 32
                )
            ),
            1,
        )
        schedule = self.task_config.env_config.get("staged_curriculum", {}).get(
            "recovery_reset_schedule", ()
        )
        maximum = max(len(schedule) - 1, 0)
        for success in results:
            self._recovery_curriculum_results.append(success)
            if len(self._recovery_curriculum_results) < window:
                continue
            success_rate = sum(self._recovery_curriculum_results) / window
            self._recovery_curriculum_results.clear()
            self._recovery_curriculum_passes = (
                self._recovery_curriculum_passes + 1 if success_rate >= 0.75 else 0
            )
            if (
                self._recovery_curriculum_passes >= 2
                and self._recovery_curriculum_level < maximum
            ):
                self._recovery_curriculum_level += 1
                self._recovery_curriculum_passes = 0

    def _recovery_curriculum_enabled(self) -> bool:
        return bool(
            self.task_config.env_config.get("staged_curriculum", {}).get(
                "recovery_curriculum", True
            )
        )

    def _range_draw(
        self,
        ids: torch.Tensor,
        key: str,
        default: tuple[float, float],
        *,
        family: int | None = None,
        width: int = 1,
        normal: bool = False,
        center: float | None = None,
        correlated: bool = False,
    ) -> torch.Tensor:
        dr = self.task_config.env_config.get("domain_rand", {})
        start_end = {
            "friction": ("friction_start_range", "friction_range"),
            "restitution": ("restitution_start_range", "restitution_range"),
            "payload": ("added_mass_start_range", "added_mass_range"),
            "link_mass": (
                "multiplied_link_mass_start_range",
                "multiplied_link_mass_range",
            ),
            "com_xy": ("added_base_com_start_range", "added_base_com_range"),
            "com_z": ("added_base_com_start_range", "added_base_com_range"),
            "motor": ("motor_strength_start_range", "motor_strength_range"),
            "pd": ("stiffness_multiplier_start_range", "stiffness_multiplier_range"),
            "damping": ("damping_multiplier_start_range", "damping_multiplier_range"),
            "zero": ("motor_zero_offset_start_range", "motor_zero_offset_range"),
        }[key]
        low_cfg = tuple((float(x) for x in dr.get(start_end[0], default)))
        high_cfg = tuple((float(x) for x in dr.get(start_end[1], default)))
        iteration = self.training_iteration
        progress = robot_dr_progress(dr, iteration)
        low = torch.full(
            (len(ids),),
            low_cfg[0] + progress * (high_cfg[0] - low_cfg[0]),
            device=self.device,
        )
        high = torch.full(
            (len(ids),),
            low_cfg[1] + progress * (high_cfg[1] - low_cfg[1]),
            device=self.device,
        )
        if normal:
            mean = (
                (low + high) * 0.5 if center is None else torch.full_like(low, center)
            )
            sigma = (high - low).clamp_min(1e-06) / 4.0
            root2 = math.sqrt(2.0)
            cdf = lambda x: 0.5 * (1.0 + torch.erf(x / root2))
            lower, upper = (cdf((low - mean) / sigma), cdf((high - mean) / sigma))
            p = lower[:, None] + (upper - lower)[:, None] * torch.rand(
                len(ids), width, device=self.device, generator=self._rng
            )
            if correlated and width > 1:
                shared = torch.rand(
                    len(ids), 1, device=self.device, generator=self._rng
                )
                local = torch.rand(
                    len(ids), width, device=self.device, generator=self._rng
                )
                p = lower[:, None] + (upper - lower)[:, None] * (
                    0.8 * shared + 0.2 * local
                )
            return (
                mean[:, None]
                + sigma[:, None]
                * root2
                * torch.erfinv(2.0 * p.clamp(1e-06, 1 - 1e-06) - 1.0)
            ).clamp(min=low[:, None], max=high[:, None])
        return low[:, None] + (high - low)[:, None] * torch.rand(
            len(ids), width, device=self.device, generator=self._rng
        )

    def initialize_structure(self) -> None:
        """Apply per-world mass/COM randomization once after model expansion."""
        if self._structural_dr_initialized:
            return
        self.initialize_terrain_assignment()
        ids = torch.arange(self.num_envs, device=self.device)
        mass = self._range_draw(
            ids, "link_mass", (0.95, 1.05), width=len(self._robot_body_ids) - 1
        )
        payload = self._range_draw(ids, "payload", (0.0, 0.0))
        com_xy = self._range_draw(ids, "com_xy", (0.0, 0.0), width=2)
        com_z = self._range_draw(ids, "com_z", (0.0, 0.0))
        self.link_mass_ratio.fill_(1.0)
        non_base_local_ids = [
            index
            for index in range(len(self._robot_body_ids))
            if index != self._base_body_local_id
        ]
        self.link_mass_ratio[:, non_base_local_ids] = mass
        self.payload_mass.copy_(payload)
        self.base_com_offset[:, :2].copy_(com_xy)
        self.base_com_offset[:, 2:3].copy_(com_z)
        model = self.env.sim.model
        baseline_mass = self._model_baseline.get("body_mass")
        baseline_inertia = self._model_baseline.get("body_inertia")
        baseline_ipos = self._model_baseline.get("body_ipos")
        if baseline_mass is not None:
            if baseline_mass.ndim != 2 or baseline_mass.shape[0] != self.num_envs:
                raise ValueError(
                    f"MuJoCo-Warp body_mass is not per-environment: {tuple(baseline_mass.shape)}"
                )
            body_ids = self._robot_body_ids
            baseline = baseline_mass.clone()
            mass_all = self.link_mass_ratio.new_ones(self.num_envs, baseline.shape[1])
            mass_all[:, body_ids] = self.link_mass_ratio
            model.body_mass.copy_(baseline * mass_all)
            base_id = self._robot_body_ids[self._base_body_local_id]
            model.body_mass[:, base_id] += self.payload_mass[:, 0]
            if baseline_inertia is not None:
                if (
                    baseline_inertia.ndim != 3
                    or baseline_inertia.shape[:2] != baseline_mass.shape
                ):
                    raise ValueError(
                        f"MuJoCo-Warp body_inertia has incompatible shape: {tuple(baseline_inertia.shape)}"
                    )
                inertia_scale = torch.ones_like(baseline_mass)
                inertia_scale[:, body_ids] = self.link_mass_ratio
                inertia_scale[:, base_id] = model.body_mass[:, base_id] / baseline_mass[
                    :, base_id
                ].clamp_min(1e-08)
                model.body_inertia.copy_(baseline_inertia * inertia_scale.unsqueeze(-1))
        if baseline_ipos is not None:
            if baseline_ipos.ndim != 3 or baseline_ipos.shape[0] != self.num_envs:
                raise ValueError(
                    f"MuJoCo-Warp body_ipos is not per-environment: {tuple(baseline_ipos.shape)}"
                )
            model.body_ipos.copy_(baseline_ipos)
            base_id = self._robot_body_ids[self._base_body_local_id]
            model.body_ipos[:, base_id] += self.base_com_offset
        self._structural_dr_initialized = True

    def randomize_episode(self, ids: torch.Tensor) -> None:
        self.initialize_structure()
        friction = self._range_draw(ids, "friction", (1.0, 1.0))
        restitution = self._range_draw(ids, "restitution", (0.0, 0.0))
        motor = self._range_draw(ids, "motor", (1.0, 1.0), width=12)
        stiffness = self._range_draw(ids, "pd", (1.0, 1.0), width=12)
        damping = self._range_draw(ids, "damping", (1.0, 1.0), width=12)
        zero = self._range_draw(ids, "zero", (0.0, 0.0), width=12)
        dr = self.task_config.env_config.get("domain_rand", {})
        low_delay = int(dr.get("action_delay_max_steps_start", 0))
        high_delay = int(dr.get("action_delay_max_steps_end", 0))
        p = robot_dr_progress(dr, self.training_iteration)
        limit = round(low_delay + p * (high_delay - low_delay))
        delay = torch.zeros(len(ids), dtype=torch.long, device=self.device)
        max_delay = torch.full_like(delay, max(limit, 0))
        self.friction[ids] = friction
        self.restitution[ids] = restitution
        self.motor_strength[ids] = motor
        self.pd_stiffness[ids] = stiffness
        self.pd_damping[ids] = damping
        self.motor_zero_offsets[ids] = zero
        self._assign_delay(ids, delay, max_delay)
        self._write_episode_domain_model(ids, include_contact=True)
        self._domain_initialized = False

    def _assign_delay(
        self, ids: torch.Tensor, steps: torch.Tensor, max_steps: torch.Tensor
    ) -> None:
        factor = round(
            float(self.task_config.env_config["sim"]["dt"]) / self.physics_dt
        )
        steps, max_steps = (steps * factor, max_steps * factor)
        self.action_delay_steps[ids] = steps
        self.action_delay_max_steps[ids] = max_steps
        action_term = self.env.action_manager.get_term("joint_pos")
        action_term.action_delay_steps[ids] = steps
        action_term.motor_zero_offsets[ids] = self.motor_zero_offsets[ids]

    def _write_episode_domain_model(
        self, ids: torch.Tensor, *, include_contact: bool = True
    ) -> None:
        model = self.env.sim.model
        if include_contact and "geom_friction" in self._model_baseline:
            baseline = self._model_baseline["geom_friction"]
            model.geom_friction[ids] = baseline[ids]
            mask = self._robot_geom_mask
            model.geom_friction[ids[:, None], mask[None, :], 0] = self.friction[ids]
        if include_contact and "geom_solref" in self._model_baseline:
            baseline = self._model_baseline["geom_solref"]
            model.geom_solref[ids] = baseline[ids]
            e = self.restitution[ids].clamp(0.0, 0.95)
            loge = torch.log(e.clamp_min(1e-05))
            damping_ratio = (-loge / torch.sqrt(math.pi**2 + loge.square())).clamp(
                0.15, 1.0
            )
            model.geom_solref[ids[:, None], self._robot_geom_mask[None, :], 1] = (
                damping_ratio
            )
        if "actuator_gainprm" in self._model_baseline:
            actuator_ids = torch.as_tensor(
                self.entity.indexing.actuator_ids, device=self.device, dtype=torch.long
            )
            model.actuator_gainprm[ids] = self._model_baseline["actuator_gainprm"][ids]
            model.actuator_biasprm[ids] = self._model_baseline["actuator_biasprm"][ids]
            model.actuator_forcerange[ids] = self._model_baseline[
                "actuator_forcerange"
            ][ids]
            kp_scale = self.pd_stiffness[ids] * self.motor_strength[ids]
            kd_scale = self.pd_damping[ids] * self.motor_strength[ids]
            model.actuator_gainprm[ids[:, None], actuator_ids[None, :], 0] *= kp_scale
            model.actuator_biasprm[ids[:, None], actuator_ids[None, :], 1] *= kp_scale
            model.actuator_biasprm[ids[:, None], actuator_ids[None, :], 2] *= kd_scale
            model.actuator_forcerange[
                ids[:, None], actuator_ids[None, :], :
            ] *= self.motor_strength[ids, :, None]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = _event_env_ids(self.env, env_ids)
        if not self._structural_dr_initialized:
            self.initialize_structure()
        self._record_recovery_outcomes(ids)
        self.randomize_episode(ids)
        terrain = self.env.scene.terrain
        if (
            terrain is not None
            and terrain.terrain_origins is not None
            and (not getattr(self, "fixed_reset_origins", False))
        ):
            terrain.update_env_origins(
                ids,
                self.episode_success[ids].to(torch.bool),
                self.episode_failure[ids].to(torch.bool),
            )
            if terrain.env_origins.data_ptr() != self.env.scene.env_origins.data_ptr():
                self.env.scene.env_origins[ids] = terrain.env_origins[ids]
        self.episode_success[ids] = False
        self.episode_failure[ids] = False
        self._step_ray_data = None
        self.recovery_reset_type[ids] = 0
        self.recovery_reward_active[ids] = False
        self.recovery_target_command[ids] = 0.0
        recovery = self._recovery_pool[ids].clone()
        recovery_positions = recovery.nonzero(as_tuple=False).flatten()
        recovery_ids = ids[recovery_positions]
        schedule = self.task_config.env_config.get("staged_curriculum", {}).get(
            "recovery_reset_schedule", ()
        )
        hard_fraction = (
            float(schedule[min(self._recovery_curriculum_level, len(schedule) - 1)][2])
            if schedule
            else 0.4
        )
        if recovery_ids.numel():
            draw = torch.rand(
                len(recovery_ids), device=self.device, generator=self._rng
            )
            self.recovery_reset_type[recovery_ids] = torch.where(
                draw < 1.0 - hard_fraction, 1, 2
            )
            self._recovery_reset_level[recovery_ids] = self._recovery_curriculum_level
        self._recovery_reset_level[ids[~recovery]] = -1
        state = self.entity.data
        origins = self.env.scene.env_origins[ids]
        root = state.default_root_state[ids].clone()
        root[:, :3] += origins
        default_joint_pos = state.default_joint_pos[ids]
        joints = default_joint_pos * (
            0.5 + torch.rand(len(ids), 12, device=self.device, generator=self._rng)
        )
        joint_vel = torch.zeros_like(joints)
        yaw = (
            2.0 * torch.rand(len(ids), device=self.device, generator=self._rng) - 1.0
        ) * math.pi
        root[:, 3:7] = 0.0
        root[:, 3] = torch.cos(yaw * 0.5)
        root[:, 6] = torch.sin(yaw * 0.5)
        root[:, :2] += (
            2.0 * torch.rand(len(ids), 2, device=self.device, generator=self._rng) - 1.0
        )
        root[:, 7:13] = (
            torch.rand(len(ids), 6, device=self.device, generator=self._rng) - 0.5
        )
        init_cfg = self.task_config.env_config.get("init_state", {})
        reference_reset = bool(
            init_cfg.get("amp_reference_state_initialization", False)
        )
        if reference_reset:
            if self._reference_motion_loader is None:
                from seesaw_policy.rl_utils.datasets import AMPLoader

                self._reference_motion_loader = AMPLoader(
                    self.device,
                    self.dt,
                    motion_files=init_cfg["s1_reference_motion_files"],
                    preload_transitions=False,
                )
            frames = self._reference_motion_loader.get_full_frame_batch(len(ids))
            root, joints, joint_vel = reference_frame_to_reset_state(frames, origins)
        if len(recovery_ids):
            bank = self._recovery_bank
            if bank is None:
                raise RuntimeError("Recovery bank has not loaded")
            row = torch.empty(len(recovery_ids), dtype=torch.long)
            recovery_types_cpu = self.recovery_reset_type[recovery_ids].detach().cpu()
            indices = bank.get("indices")
            for reset_type, label in ((1, 1), (2, 2)):
                selected = recovery_types_cpu == reset_type
                if not selected.any():
                    continue
                pool = (
                    indices[label]
                    if indices is not None
                    else torch.arange(bank["root_states"].shape[0])
                )
                choice = torch.randint(
                    len(pool), (int(selected.sum()),), generator=self._recovery_rng
                )
                row[selected] = pool[choice]
            sampled_root = bank["root_states"][row].to(self.device)
            root_index = recovery_positions
            root[root_index, :3] = (
                sampled_root[:, :3] + self.env.scene.env_origins[recovery_ids]
            )
            root[root_index, 3:7] = _xyzw_to_wxyz(sampled_root[:, 3:7])
            root[root_index, 7:13] = sampled_root[:, 7:13]
            joints[root_index] = bank["dof_pos"][row].to(self.device)
            joint_vel[root_index] = bank["dof_vel"][row].to(self.device)
        command = self.env.command_manager.get_term("twist").command
        command[ids] = 0.0
        self.entity.write_root_state_to_sim(root, env_ids=ids)
        self.entity.write_joint_state_to_sim(joints, joint_vel, env_ids=ids)
        self.env.sim.data.xfrc_applied[ids] = 0.0
        self.disturbance_wrench_timer[ids] = 0.0
        self.external_wrenches[ids] = 0.0
        self.recovery_timer[ids] = 0.0
        self.recovery_success_hold[ids] = 0.0
        self._recovery_success[ids] = False
        self._recovery_failure[ids] = False
        self.route_max_x[ids] = root[:, 0]
        self.route_progress_check_x[ids] = root[:, 0]
        self.route_no_progress_s[ids] = 0.0
        self.route_progress_event[ids] = 0.0
        self.route_stalled[ids] = False
        self.route_backtracked[ids] = False
        self.route_heading_bad[ids] = False
        self.route_finished[ids] = False
        self.route_failed[ids] = False
        self.route_overhead_jump[ids] = False
        self.last_task_termination[ids] = False
        self.contact_duration[ids] = 0.0
        self.air_duration[ids] = 0.0
        self.airtime_reward[ids] = 0.0
        self.contact_ema[ids] = 0.0
        self.previous_joint_vel[ids] = joint_vel
        self.previous_base_lin_vel_b[ids] = quat_apply_inverse(
            root[:, 3:7], root[:, 7:10]
        )
        self.previous_base_ang_vel_b[ids] = quat_apply_inverse(
            root[:, 3:7], root[:, 10:13]
        )
        self.previous_root_lin_vel_w[ids] = root[:, 7:10]
        self.previous_root_ang_vel_w[ids] = root[:, 10:13]
        self._history_initialized[ids] = True
        self.recovery_start_height[ids] = root[:, 2] - origins[:, 2]
        rpy = _quat_to_rpy_wxyz(root[:, 3:7])
        self.recovery_start_tilt[ids] = torch.linalg.vector_norm(rpy[:, :2], dim=1)
        upright = (rpy[:, 0].cos() * rpy[:, 1].cos()).clamp(-1.0, 1.0)
        recovery_tilt = torch.acos(upright)
        orient_gate = ((upright - 0.5) / (math.cos(math.radians(20.0)) - 0.5)).clamp(
            0.0, 1.0
        )
        self.recovery_prev_height[ids] = orient_gate * (
            self.recovery_start_height[ids] / 0.35
        ).clamp(0.0, 1.0)
        self.recovery_prev_tilt[ids] = 1.0 - recovery_tilt / math.pi
        self.recovery_origin_xy[ids] = root[:, :2]
        self.push_grace_timer[ids] = float(
            self.task_config.env_config["domain_rand"]["push_recovery_grace_s"]
            if "robot_dr_ramp" in self.task_config.env_config.get("domain_rand", {})
            else 2.0
        )
        self.push_next_event[ids] = 3.0 + 3.0 * torch.rand(
            len(ids), device=self.device, generator=self._rng
        )
        self.push_was_recovery[ids] = False
        action_term = self.env.action_manager.get_term("joint_pos")
        action_term.action_delay_steps[ids] = self.action_delay_steps[ids]
        action_term.motor_zero_offsets[ids] = self.motor_zero_offsets[ids]

    def update_training_iteration(self, iteration: int) -> None:
        self.training_iteration = int(iteration)

    def _base_height_from_rays(
        self, ray_data: dict[str, Any], root_z: torch.Tensor
    ) -> torch.Tensor:
        """Use current local ground; a missing ground scan is explicitly invalid."""
        from .ground_height import local_base_height

        del root_z
        height, self.base_height_valid = local_base_height(
            ray_data["heightmap"],
            self._base_height_x,
            self._base_height_y,
            center_surface=False,
        )
        return height

    def _tracking_velocity(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.task_config.env_config.get("rewards", {}).get(
            "heading_velocity_tracking", False
        ):
            return (data.root_link_lin_vel_b[:, :2], data.root_link_ang_vel_b[:, 2])
        yaw = _quat_to_rpy_wxyz(data.root_link_quat_w)[:, 2]
        cosine, sine = (yaw.cos(), yaw.sin())
        world = data.root_link_lin_vel_w
        return (
            torch.stack(
                (
                    cosine * world[:, 0] + sine * world[:, 1],
                    -sine * world[:, 0] + cosine * world[:, 1],
                ),
                dim=1,
            ),
            data.root_link_ang_vel_w[:, 2],
        )

    def _tracking_sigmas(self, command: torch.Tensor) -> torch.Tensor:
        """Equivalent squared-error denominators for relative x/y/yaw tracking.

        Nonzero commands use error / max(abs(target), 0.1). Exactly zero
        commands retain absolute-error tracking instead of dividing by zero.
        Raw errors remain physical errors for curriculum and diagnostics.
        """
        rewards = self.task_config.env_config.get("rewards", {})
        sigma = float(rewards.get("tracking_sigma", 0.1))
        if not rewards.get("tracking_relative_error", False):
            return torch.full_like(command[:, :3], sigma)
        targets = command[:, :3].abs()
        floor = float(rewards["tracking_relative_target_floor"])
        denominator = torch.where(
            targets > 0.0, targets.clamp_min(floor), torch.ones_like(targets)
        )
        return sigma * denominator.square()

    def _state(
        self, *, refresh_rays: bool = True
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        env, entity = (self.env, self.entity)
        data = entity.data
        n = self.num_envs
        zero = lambda width: torch.zeros(
            n, width, dtype=torch.float32, device=self.device
        )
        root_pos = data.root_link_pos_w
        root_quat = data.root_link_quat_w
        base_lin = data.root_link_lin_vel_b
        base_ang = data.root_link_ang_vel_b
        root_lin = data.root_link_lin_vel_w
        root_ang = data.root_link_ang_vel_w
        action = env.action_manager.action
        command = env.command_manager.get_command("twist")
        tracking_lin, tracking_yaw = self._tracking_velocity(data)
        tracking_sigmas = self._tracking_sigmas(command)
        shared = getattr(self, "_physical_state_cache", None)
        if shared is not None and "body" in shared:
            (
                feet_pos,
                feet_vel,
                feet_pos_base,
                feet_force,
                body_force_all,
                body_force,
                contact,
            ) = shared["body"]
        else:
            feet_pos = data.body_link_pos_w[:, self.feet_ids]
            feet_vel = data.body_link_lin_vel_w[:, self.feet_ids]
            feet_pos_base = quat_apply_inverse(
                root_quat[:, None, :].expand(-1, len(self.feet_ids), -1).reshape(-1, 4),
                (feet_pos - root_pos[:, None, :]).reshape(-1, 3),
            ).reshape(n, len(self.feet_ids), 3)
            feet_sensor = env.scene["feet_ground_contact"].data
            body_sensor = env.scene["body_ground_contact"].data
            feet_force = -feet_sensor.force
            body_force_all = separate_foot_geom_contact_forces(
                -body_sensor.force, feet_force, self._calf_body_ids, self._foot_body_ids
            )
            body_force = body_force_all[:, self._body_contact_ids]
            contact = feet_force[..., 2] > 1.0
            if shared is not None:
                shared["body"] = (
                    feet_pos,
                    feet_vel,
                    feet_pos_base,
                    feet_force,
                    body_force_all,
                    body_force,
                    contact,
                )
        action_term = env.action_manager.get_term("joint_pos")
        executed = action_term.executed_target
        if refresh_rays:
            if self._step_ray_data is None:
                self.env.sim.sense()
                ray_data, _ = self.rays.read(pack=False)
            else:
                ray_data = self._step_ray_data
                self._step_ray_data = None
        else:
            ray_data = env.terrain_data.data
        if shared is not None and "terrain" in shared:
            height_map, foot_rel, foot_valid, feet_ground_z, roughness = shared[
                "terrain"
            ]
        else:
            height_map = ray_data["heightmap"]["height"].flatten(1)
            footmap = ray_data["footmap"]
            foot_size = len(
                self.task_config.train_config["policy"]["critic_terrain_schema"][
                    "foot_axis"
                ]
            )
            foot_rel = footmap["height"]
            foot_valid = footmap["valid"]
            axis = self.task_config.train_config["policy"]["critic_terrain_schema"][
                "foot_axis"
            ]
            selected = [
                int(np.argmin(np.abs(np.asarray(axis) - x))) for x in (-0.05, 0.0, 0.05)
            ]
            selected_heights = foot_rel[:, :, selected][:, :, :, selected].reshape(
                n, 4, 9
            )
            selected_valid = foot_valid[:, :, selected][:, :, :, selected].reshape(
                n, 4, 9
            )
            feet_ground_z = torch.where(
                selected_valid,
                feet_pos[:, :, 2:3] + selected_heights,
                feet_pos[:, :, 2:3].expand(-1, -1, 9),
            ).reshape(n, 4, 9)
            roughness = self._foothold_plane_residual(foot_rel, foot_valid)
            if shared is not None:
                shared["terrain"] = (
                    height_map,
                    foot_rel,
                    foot_valid,
                    feet_ground_z,
                    roughness,
                )
        base_origin_z = env.scene.env_origins[:, 2]
        base_height = self._base_height_from_rays(ray_data, root_pos[:, 2])
        rpy = _quat_to_rpy_wxyz(root_quat)
        body_mass_field = env.sim.model.body_mass
        if body_mass_field.ndim == 2:
            body_masses = body_mass_field[:, self._robot_body_ids]
        else:
            body_masses = self._model_baseline["body_mass"][:, self._robot_body_ids]
        friction = self.friction
        geom_solref = env.sim.model.geom_solref
        if geom_solref.ndim == 3 and self._robot_geom_mask is not None:
            restitution_raw = self.restitution
        else:
            restitution_raw = zero(1)
        action_fifo_term = action_term.action_fifo
        spatial_priv = ray_data["spatial_privileged"]
        feet_contact = contact.to(torch.float32)
        external = {
            "upright": (1.0 - data.projected_gravity_b[:, 2]).clamp_min(0.0) * 0.5,
            "feet_stair_face": self._stair_face_score(feet_force, foot_rel, foot_valid),
            "foot_contact_balance": self._contact_balance_score(contact),
            "parkour_ground_obstacle_contact": self._ground_obstacle_contact(
                feet_pos, contact
            ),
            "parkour_pillar_collision": self._pillar_collision(body_force),
        }
        upright = (-data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
        recovery_tilt = torch.acos(upright)
        recovery_flip_potential = 1.0 - recovery_tilt / math.pi
        recovery_height_gate = (
            (upright - 0.5) / (math.cos(math.radians(20.0)) - 0.5)
        ).clamp(0.0, 1.0)
        recovery_height_potential = recovery_height_gate * (base_height / 0.35).clamp(
            0.0, 1.0
        )
        total_mass = body_masses.sum(1).clamp_min(1.0)
        foot_compact = (
            (torch.linalg.vector_norm(feet_pos - root_pos[:, None, :], dim=-1) - 0.2)
            .clamp_min(0.0)
            .mean(1)
        )
        if bool(
            self.task_config.env_config.get("rewards", {}).get(
                "recovery_compact_tilt_gate", False
            )
        ):
            compact_gate = ((recovery_tilt - math.pi / 6.0) / (math.pi / 3.0)).clamp(
                0.0, 1.0
            )
            foot_compact = foot_compact * compact_gate
        recovery_cfg = self.task_config.env_config.get("staged_curriculum", {})
        stable_hold = float(recovery_cfg.get("recovery_stable_hold_s", 0.0))
        outcome_failure_ratio = 2.0 / 3.0 if stable_hold > 0.0 else 1.0
        external.update(
            {
                "recovery_flip": recovery_flip_potential - self.recovery_prev_tilt,
                "recovery_height": (base_height >= 0.35).float() * recovery_height_gate,
                "recovery_height_progress": recovery_height_potential
                - self.recovery_prev_height,
                "recovery_upright": 1.0 - recovery_tilt / math.pi,
                "recovery_position": torch.expm1(
                    -(root_pos[:, :2] - self.recovery_origin_xy).square().sum(1)
                    / 0.5**2
                ),
                "recovery_tilt_cost": -recovery_tilt / math.pi,
                "recovery_height_cost": recovery_height_potential - 1.0,
                "recovery_outcome": self._recovery_success.float()
                - outcome_failure_ratio * self._recovery_failure.float(),
                "recovery_contact": body_force.norm(dim=-1).sum(1)
                / (total_mass * 9.81),
                "recovery_feet_compact": foot_compact,
            }
        )
        state = {
            "base_lin_vel": base_lin,
            "base_ang_vel": base_ang,
            "last_base_lin_vel": self.previous_base_lin_vel_b,
            "last_base_ang_vel": self.previous_base_ang_vel_b,
            "root_lin_vel_w": root_lin,
            "root_ang_vel_w": root_ang,
            "last_root_lin_vel_w": self.previous_root_lin_vel_w,
            "last_root_ang_vel_w": self.previous_root_ang_vel_w,
            "projected_gravity": data.projected_gravity_b,
            "commands": command,
            "joint_pos": data.joint_pos,
            "default_joint_pos": data.default_joint_pos,
            "joint_vel": data.joint_vel,
            "joint_pos_limits": data.joint_pos_limits,
            "joint_vel_limits": self.joint_velocity_limits.expand(n, -1),
            "last_joint_vel": self.previous_joint_vel,
            "actions": action,
            "last_actions": env.action_manager.prev_action,
            "last_last_actions": env.action_manager.prev_prev_action,
            "previous_actions": env.action_manager.prev_action,
            "torques": data.actuator_force,
            "torque_limits": self.torque_limits.expand(n, -1),
            "root_pos_world": root_pos,
            "root_quat_wxyz": root_quat,
            "feet_pos_world": feet_pos,
            "feet_pos_base": feet_pos_base,
            "feet_vel_world": feet_vel,
            "feet_contact_force_world": feet_force,
            "feet_terrain_heights": feet_ground_z,
            "feet_roughness": roughness,
            "body_contact_force_world": body_force,
            "terrain_height": height_map,
            "motor_strength": self.motor_strength,
            "pd_stiffness": self.pd_stiffness,
            "pd_damping": self.pd_damping,
            "motor_zero_offsets": self.motor_zero_offsets,
            "action_delay": self._critic_action_delay(),
            "body_masses": body_masses,
            "restitution": restitution_raw,
            "payload_mass": self.payload_mass,
            "base_com_offset": self.base_com_offset,
            "friction": friction,
            "recovery_state": self._recovery_state(),
            "recovery_reward_active": self.recovery_reward_active,
            "velocity_impulse_world": self.velocity_impulse_world,
            "external_force_world": self.external_wrenches[:, :3],
            "external_torque_world": self.external_wrenches[:, 3:],
            "spatial_rays": spatial_priv,
            "executed_target": executed,
            "action_fifo": action_fifo_term,
            "contact_duration": self.contact_duration,
            "air_duration": self.air_duration,
            "episode_length": env.episode_length_buf,
            "dt": self.dt,
            "feet_contact": contact,
            "feet_slip_contact": feet_force.norm(dim=-1)
            > float(
                self.task_config.env_config["rewards"].get(
                    "feet_slip_contact_threshold", 10.0
                )
            ),
            "feet_vel_w": feet_vel,
            "feet_air_time_reward": self._gated_airtime_reward(command),
            "feet_stair_face_score": external["feet_stair_face"],
            "penalized_contact_norm": body_force[:, 1:].norm(dim=-1),
            "termination_contact_norm": body_force[:, :1].norm(dim=-1),
            "contact_force_norm": body_force_all.norm(dim=-1),
            "base_height": base_height,
            "roll": rpy[:, 0],
            "tracking_linear_error_sq": (command[:, :2] - tracking_lin).square(),
            "tracking_angular_error_sq": (command[:, 2] - tracking_yaw).square(),
            "tracking_linear_sigmas": tracking_sigmas[:, :2],
            "tracking_angular_sigma": tracking_sigmas[:, 2],
            "joint_names": self.joint_names,
            "external_terms": external,
            "terminated": self.episode_failure,
            "time_out": (
                env.reset_time_outs
                if hasattr(env, "reset_time_outs")
                else torch.zeros(n, dtype=torch.bool, device=self.device)
            ),
        }
        return (state, ray_data)

    def _stair_face_score(
        self, force: torch.Tensor, foot_rel: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(force.shape[0], device=force.device)

    def _contact_balance_score(self, contact: torch.Tensor) -> torch.Tensor:
        alpha = 1.0 - math.exp(-self.dt / 4.0)
        self.contact_ema.mul_(1.0 - alpha).add_(contact.float(), alpha=alpha)
        mean = self.contact_ema.mean(1, keepdim=True)
        return (self.contact_ema - mean).square().sum(1)

    def _ground_obstacle_contact(
        self, feet_pos: torch.Tensor, contact: torch.Tensor
    ) -> torch.Tensor:
        result = torch.zeros(self.num_envs, device=self.device)
        return result

    def _pillar_collision(self, body_force: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(self.num_envs, device=self.device)
        return result

    def _critic_action_delay(self) -> torch.Tensor:
        decimation = float(round(self.dt / self.physics_dt))
        current = self.action_delay_steps.float() / decimation
        current = torch.where(self.env.episode_length_buf == 0, 0.0, current)
        maximum = (
            self.action_delay_max_steps.amax().clamp(0, int(decimation)).float()
            / decimation
        )
        return torch.stack((current, maximum.expand_as(current)), dim=1)

    def _recovery_state(self) -> torch.Tensor:
        active = self.recovery_reset_type > 0
        config = self.task_config.env_config.get("staged_curriculum", {})
        timeout_s = max(float(config.get("recovery_timeout_s", 6.0)), 1e-06)
        hold_s = max(float(config.get("recovery_stable_hold_s", 1.0)), self.dt)
        state = torch.zeros(self.num_envs, 9, device=self.device)
        state[:, 0] = active.float()
        state[:, 1] = (1.0 - self.recovery_timer / timeout_s).clamp(0.0, 1.0)
        state[:, 2] = (self.recovery_success_hold / hold_s).clamp(0.0, 1.0)
        delta = torch.zeros(self.num_envs, 3, device=self.device)
        delta[:, :2] = self.entity.data.root_link_pos_w[:, :2] - self.recovery_origin_xy
        state[:, 3:5] = quat_apply_inverse(self.entity.data.root_link_quat_w, delta)[
            :, :2
        ]
        obs_scales = self.task_config.env_config["normalization"]["obs_scales"]
        state[:, 5:8] = self.recovery_target_command * torch.tensor(
            (
                float(obs_scales["lin_vel"]),
                float(obs_scales["lin_vel"]),
                float(obs_scales["ang_vel"]),
            ),
            device=self.device,
        )
        state[:, 1:8] *= active[:, None]
        state[:, 8] = self._recovery_pool.float()
        return state

    def actor_observation(self, env=None) -> torch.Tensor:
        del env
        state = self.entity.data
        return build_actor_observation(
            base_ang_vel=state.root_link_ang_vel_b,
            projected_gravity=state.projected_gravity_b,
            commands=self.command,
            joint_pos=state.joint_pos,
            default_joint_pos=state.default_joint_pos,
            joint_vel=state.joint_vel,
            actions=self.env.action_manager.action,
            recipe=self.recipe_view,
        )

    def critic_observation(self, env=None) -> torch.Tensor:
        del env
        state, ray_data = self._state()
        self.env.terrain_data.data = ray_data
        return self._build_critic(state)

    def _build_critic(self, state):
        return build_critic_observation(
            build_critic_fields(state, self.recipe_view), self.recipe_view
        )

    def amp_observation(self) -> torch.Tensor:
        state = self.entity.data
        observation = build_amp_observation(
            joint_pos=state.joint_pos,
            feet_pos_world=state.body_link_pos_w[:, self.feet_ids],
            root_pos_world=state.root_link_pos_w,
            root_quat_wxyz=state.root_link_quat_w,
            base_lin_vel=state.root_link_lin_vel_b,
            base_ang_vel=state.root_link_ang_vel_b,
            joint_vel=state.joint_vel,
            recipe=self.recipe_view,
        )
        if self.task_config.env_config["env"].get("amp_mask_base_height", False):
            observation[:, -1] = 0.0
            self.amp_height_valid.fill_(True)
        return observation

    def read_terrain(
        self, *, pack: bool = True
    ) -> tuple[dict[str, Any], torch.Tensor | None]:
        data, packed = self.rays.read(purpose="export", pack=pack)
        self.env.terrain_data.data = data
        return (data, packed)

    def termination_measurements(self):
        """Physical checks shared by controllers of the same simulation scene."""
        state = self.entity.data
        body_force = self.env.scene["body_ground_contact"].data.force
        trunk_force = self.env.scene["trunk_ground_contact"].data.force
        return {
            "rpy": _quat_to_rpy_wxyz(state.root_link_quat_w),
            "upright": -state.projected_gravity_b[:, 2],
            "base_force": trunk_force[:, 0].norm(dim=1),
            "lin_speed": state.root_link_lin_vel_w.norm(dim=1),
            "ang_speed": state.root_link_ang_vel_w.norm(dim=1),
            "finite": finite_worlds(
                state.root_link_pos_w,
                state.root_link_quat_w,
                state.root_link_lin_vel_w,
                state.root_link_ang_vel_w,
                state.joint_pos,
                state.joint_vel,
                state.actuator_force,
                self.env.sim.data.qpos,
                self.env.sim.data.qvel,
                self.env.sim.data.qacc,
                self.env.sim.data.qacc_warmstart,
                self.env.scene["feet_ground_contact"].data.force,
                body_force,
                trunk_force,
            ),
        }

    def task_termination(self, env=None, *, measurements=None) -> torch.Tensor:
        del env
        self.route_progress_event.zero_()
        self.route_stalled.zero_()
        self.route_backtracked.zero_()
        self.route_heading_bad.zero_()
        self.route_finished.zero_()
        self.route_failed.zero_()
        self.route_overhead_jump.zero_()
        self._recovery_failure.zero_()
        state = self.entity.data
        origin_z = self.env.scene.env_origins[:, 2]
        if self._step_ray_data is None:
            self.env.sim.sense()
            self._step_ray_data, _ = self.rays.read(pack=False)
        height = self._base_height_from_rays(
            self._step_ray_data, state.root_link_pos_w[:, 2]
        )
        checks = (
            self.termination_measurements() if measurements is None else measurements
        )
        rpy = checks["rpy"]
        upright = checks["upright"]
        contact_limit = float(
            self.task_config.env_config.get("env", {}).get(
                "termination_contact_force", 10.0
            )
        )
        base_contact = checks["base_force"] > contact_limit
        tilt_limit = float(
            self.task_config.env_config.get("env", {}).get(
                "max_root_tilt_reset", math.radians(75.0)
            )
        )
        tilt_failure = rpy[:, 0].abs() > tilt_limit
        finite = checks["finite"]
        env_cfg = self.task_config.env_config.get("env", {})
        over_speed = (
            checks["lin_speed"] > float(env_cfg.get("max_root_lin_vel_reset", 8.0))
        ) | (checks["ang_speed"] > float(env_cfg.get("max_root_ang_vel_reset", 20.0)))
        recovery = self.recovery_reset_type > 0
        self.recovery_reward_active.copy_(recovery)
        good_recovery_pose = (
            height
            >= float(
                self.task_config.env_config.get("staged_curriculum", {}).get(
                    "recovery_success_height", 0.28
                )
            )
        ) & (
            upright
            >= math.cos(
                math.radians(
                    float(
                        self.task_config.env_config.get("staged_curriculum", {}).get(
                            "recovery_success_tilt_deg", 30.0
                        )
                    )
                )
            )
        )
        self.recovery_timer[recovery] += self.dt
        recovery_cfg = self.task_config.env_config.get("staged_curriculum", {})
        hold_s = max(float(recovery_cfg.get("recovery_stable_hold_s", 0.0)), 0.0)
        if hold_s > 0.0:
            lin_limit = float(
                recovery_cfg.get("recovery_stable_lin_speed", float("inf"))
            )
            ang_limit = float(
                recovery_cfg.get("recovery_stable_ang_speed", float("inf"))
            )
            good_recovery_pose &= state.root_link_lin_vel_b.norm(dim=1) <= lin_limit
            good_recovery_pose &= state.root_link_ang_vel_b.norm(dim=1) <= ang_limit
            joint_limit = recovery_cfg.get("recovery_stable_joint_rmse", 0.3)
            if joint_limit is not None:
                joint_rmse = (
                    (state.joint_pos - state.default_joint_pos).square().mean(1).sqrt()
                )
                good_recovery_pose &= joint_rmse <= float(joint_limit)
            self.recovery_success_hold.copy_(
                torch.where(
                    recovery & good_recovery_pose,
                    self.recovery_success_hold + self.dt,
                    torch.where(
                        recovery,
                        torch.zeros_like(self.recovery_success_hold),
                        self.recovery_success_hold,
                    ),
                )
            )
            success_now = (
                recovery
                & good_recovery_pose
                & (self.recovery_success_hold >= max(hold_s - 1e-06, 0.0))
            )
        else:
            success_now = recovery & good_recovery_pose
        self._recovery_success.logical_or_(success_now)
        timeout_s = float(recovery_cfg.get("recovery_timeout_s", 5.0))
        failure_now = recovery & ~success_now & (self.recovery_timer >= timeout_s)
        self._recovery_failure.logical_or_(failure_now)
        self.recovery_reset_type[success_now] = 0
        self.command[success_now] = self.recovery_target_command[success_now]
        command_term = self.env.command_manager.get_term("twist")
        command_term.time_left[success_now] = float(
            command_term.cfg.resampling_time_range[0]
        )
        low_height = height < 0.15
        failed = ~finite | ~recovery & (
            base_contact
            | tilt_failure
            | over_speed
            | (upright < math.cos(math.radians(70.0)))
            | low_height
        )
        done = failed
        self.episode_failure |= failed
        time_out = self.env.episode_length_buf >= self.env.max_episode_length - 1
        self.episode_success |= time_out & ~failed & ~recovery
        self.last_base_contact_event.copy_(base_contact & ~recovery)
        self.last_orientation_event.copy_(tilt_failure & ~recovery)
        self.last_velocity_event.copy_(over_speed & ~recovery | ~finite)
        self.last_velocity_linear_event.copy_(
            (checks["lin_speed"] > float(env_cfg.get("max_root_lin_vel_reset", 8.0)))
            & ~recovery
        )
        self.last_velocity_angular_event.copy_(
            (checks["ang_speed"] > float(env_cfg.get("max_root_ang_vel_reset", 20.0)))
            & ~recovery
        )
        self.last_velocity_nonfinite_event.copy_(~finite)
        self.last_fall_event.copy_(
            (base_contact | tilt_failure | (height < 0.15)) & ~recovery
        )
        self.last_recovery_timeout_event.copy_(self._recovery_failure)
        done |= self._recovery_failure
        self.episode_failure |= self._recovery_failure
        self.last_task_termination.copy_(done)
        return done

    def _gated_airtime_reward(self, command: torch.Tensor) -> torch.Tensor:
        threshold = float(
            self.task_config.env_config["rewards"].get(
                "feet_air_time_min_command_norm", 0.1
            )
        )
        moving = torch.linalg.vector_norm(command[:, :2], dim=1) > threshold
        return torch.where(moving, self.airtime_reward, 0.0)

    def _update_contact_history(self, feet_force: torch.Tensor) -> None:
        contact = feet_force[..., 2] > 1.0
        previous_air = self.air_duration.clone()
        landing = contact & (previous_air > 0.0)
        credited = (
            previous_air
            - float(
                self.task_config.env_config["rewards"].get(
                    "feet_air_time_threshold", 0.35
                )
            )
        ).clamp(
            max=float(
                self.task_config.env_config["rewards"].get("feet_air_time_cap", 0.6)
            )
        )
        self.airtime_reward.copy_((credited * landing.float()).sum(1))
        self.contact_duration.copy_(
            torch.where(contact, self.contact_duration + self.dt, 0.0)
        )
        self.air_duration.copy_(torch.where(contact, 0.0, self.air_duration + self.dt))

    def compute_reward(self, env=None) -> torch.Tensor:
        lab_env = self.env if env is None else env
        state, ray_data = self._state()
        self.last_turn_over_reward_mask.copy_(self.recovery_reward_active)
        invalid = self.last_velocity_nonfinite_event
        if bool(invalid.any()):
            import os

            count = getattr(self, "_numerical_reset_count", 0) + int(invalid.sum())
            self._numerical_reset_count = count
            if not getattr(self, "_numerical_dump_written", False):

                def selected(values):
                    return {
                        key: (
                            selected(value)
                            if isinstance(value, dict)
                            else value[invalid].detach().cpu()
                        )
                        for key, value in values.items()
                        if isinstance(value, dict)
                        or (
                            torch.is_tensor(value)
                            and value.ndim
                            and (value.shape[0] == self.num_envs)
                        )
                    }

                path = f"/tmp/mjlab_terminal_state_rank{os.environ.get('RANK', 'local')}.pt"
                torch.save(
                    {
                        "state": selected(state),
                        "env_ids": invalid.nonzero().cpu(),
                        "terrain_types": self.env.scene.terrain.terrain_types[
                            invalid
                        ].cpu(),
                        "terrain_levels": self.env.scene.terrain.terrain_levels[
                            invalid
                        ].cpu(),
                    },
                    path,
                )
                self._numerical_dump_written = True
            if count <= 10 or count % 100 == 0:
                print(
                    f"MJLAB_NUMERICAL_RESET rank={os.environ.get('RANK', 'local')} count={count} ids={invalid.nonzero().flatten().tolist()}",
                    flush=True,
                )
        batch = terminal_safe_reward(
            self.reward_model, state, invalid, self.last_task_termination
        )
        bad_reward = batch.invalid_rows
        if bad_reward is not None:
            if self.active_mask is not None:
                bad_reward = bad_reward & self.active_mask
            if bool(bad_reward.any()):
                import os

                self.last_task_termination[bad_reward] = True
                self.episode_failure[bad_reward] = True
                mark_invalid_reward_reset(lab_env, bad_reward)
                count = getattr(self, "_nonfinite_reward_reset_count", 0) + int(
                    bad_reward.sum()
                )
                self._nonfinite_reward_reset_count = count
                if count <= 10 or count % 100 == 0:
                    print(
                        f"MJLAB_NONFINITE_REWARD_RESET rank={os.environ.get('RANK', 'local')} count={count} ids={bad_reward.nonzero().flatten().tolist()}",
                        flush=True,
                    )
        batch = self._failure_reset_reward(batch)
        task = batch.task
        auxiliary = batch.auxiliary
        total = task + auxiliary
        self.last_reward_batch = RewardBatch(
            total=total,
            task=task,
            auxiliary=auxiliary,
            terms=batch.terms,
            raw_terms=batch.raw_terms,
        )
        self.last_raw_reward_terms = dict(batch.raw_terms)
        self.last_final_reward_terms = dict(batch.terms)
        self.last_tracking_reward.copy_(
            batch.terms.get("tracking_lin_vel", torch.zeros_like(task))
            + batch.terms.get("tracking_ang_vel", torch.zeros_like(task))
        )
        upright = (-self.entity.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
        tilt = torch.acos(upright)
        self.recovery_prev_tilt.copy_(1.0 - tilt / math.pi)
        base_height = state["base_height"]
        gate = ((upright - 0.5) / (math.cos(math.radians(20.0)) - 0.5)).clamp(0.0, 1.0)
        self.recovery_prev_height.copy_(gate * (base_height / 0.35).clamp(0.0, 1.0))
        self._update_contact_history(-self.env.scene["feet_ground_contact"].data.force)
        return total

    def _failure_reset_reward(self, batch: RewardBatch) -> RewardBatch:
        """Charge each failed reset once, including quarantined numerical resets."""
        failed = self.last_task_termination & self.episode_failure
        if self.active_mask is not None:
            failed = failed & self.active_mask
        penalty = failed.to(batch.task.dtype) * float(
            self.task_config.env_config["rewards"]["scales"]["termination"]
        )
        task = batch.task.clone()
        terms = dict(batch.terms)
        for name in ("termination",):
            if name in terms:
                task -= terms[name]
                terms[name] = torch.zeros_like(task)
        task += penalty
        terms["termination"] = penalty
        raw = dict(batch.raw_terms)
        raw["termination"] = failed.to(batch.task.dtype)
        return RewardBatch(
            task + batch.auxiliary,
            task,
            batch.auxiliary,
            terms,
            raw,
            batch.invalid_rows,
        )

    def finalize_step(self, dones: torch.Tensor) -> None:
        state = self.entity.data
        keep = ~dones.bool()
        self.previous_joint_vel[keep] = state.joint_vel[keep]
        self.previous_base_lin_vel_b[keep] = state.root_link_lin_vel_b[keep]
        self.previous_base_ang_vel_b[keep] = state.root_link_ang_vel_b[keep]
        self.previous_root_lin_vel_w[keep] = state.root_link_lin_vel_w[keep]
        self.previous_root_ang_vel_w[keep] = state.root_link_ang_vel_w[keep]
        self._history_initialized[keep] = True

    def push(self, env_ids: torch.Tensor | None) -> None:
        cfg = self.task_config.env_config.get("domain_rand", {})
        robustness3 = "robot_dr_ramp" in cfg
        if not robustness3:
            return
        ids = _event_env_ids(self.env, env_ids)
        if robustness3:
            ids = ids[self.recovery_reset_type[ids] == 0]
            if cfg.get("disturbance_flat_wrench_upright_only", False):
                upright = -self.entity.data.projected_gravity_b[ids, 2]
                ids = ids[
                    upright
                    >= float(cfg.get("disturbance_flat_wrench_upright_threshold", 0.7))
                ]
            if robustness3:
                age = self.env.episode_length_buf[ids] * self.task_config.control_dt
                ids = ids[
                    (age >= float(cfg["push_recovery_grace_s"]))
                    & (self.push_grace_timer[ids] <= 0.0)
                ]
            ramp_progress = (
                robot_dr_progress(cfg, self.training_iteration) if robustness3 else 0.0
            )
            participation = 0.5 + 0.5 * ramp_progress
            ids = ids[
                torch.rand(len(ids), device=self.device, generator=self._rng)
                < participation
            ]
            if not ids.numel():
                return
        interval = float(cfg.get("push_interval_s", 4.0))
        force_range = tuple(
            (
                float(x)
                for x in cfg.get(
                    "disturbance_flat_wrench_force_xy_start_range", (2.0, 12.0)
                )
            )
        )
        torque_range = tuple(
            (
                float(x)
                for x in cfg.get(
                    "disturbance_flat_wrench_torque_start_range", (0.2, 4.0)
                )
            )
        )
        duration = tuple(
            (
                float(x)
                for x in cfg.get(
                    "disturbance_flat_wrench_duration_start_range_s", (0.1, 0.25)
                )
            )
        )
        if self.training_iteration > 0:
            p = (
                robot_dr_progress(cfg, self.training_iteration)
                if robustness3
                else min(
                    self.training_iteration
                    / max(int(cfg.get("curriculum_end_iter", 20000)), 1),
                    1.0,
                )
            )
            force_range = tuple(
                (
                    a + p * (b - a)
                    for a, b in zip(
                        force_range,
                        cfg.get("disturbance_flat_wrench_force_xy_range", (10.0, 60.0)),
                        strict=True,
                    )
                )
            )
            torque_range = tuple(
                (
                    a + p * (b - a)
                    for a, b in zip(
                        torque_range,
                        cfg.get("disturbance_flat_wrench_torque_range", (1.0, 20.0)),
                        strict=True,
                    )
                )
            )
            duration = tuple(
                (
                    a + p * (b - a)
                    for a, b in zip(
                        duration,
                        cfg.get("disturbance_flat_wrench_duration_range_s", (0.2, 0.5)),
                        strict=True,
                    )
                )
            )
        force = torch.zeros(len(ids), 1, 3, device=self.device)
        torque = torch.zeros_like(force)
        if robustness3:
            force_direction = torch.randn(
                len(ids), 2, device=self.device, generator=self._rng
            )
            force_direction = force_direction / force_direction.norm(
                dim=1, keepdim=True
            ).clamp_min(1e-06)
            torque_direction = torch.randn(
                len(ids), 3, device=self.device, generator=self._rng
            )
            torque_direction = torque_direction / torque_direction.norm(
                dim=1, keepdim=True
            ).clamp_min(1e-06)
            force_size = force_range[0] + (
                force_range[1] - force_range[0]
            ) * torch.rand(len(ids), 1, device=self.device, generator=self._rng)
            torque_size = torque_range[0] + (
                torque_range[1] - torque_range[0]
            ) * torch.rand(len(ids), 1, device=self.device, generator=self._rng)
            force[:, 0, :2] = force_direction * force_size
            torque[:, 0, :] = torque_direction * torque_size
        else:
            force[:, 0, :2] = (
                2.0 * torch.rand(len(ids), 2, device=self.device, generator=self._rng)
                - 1.0
            ) * force_range[1]
            torque[:, 0, :] = (
                2.0 * torch.rand(len(ids), 3, device=self.device, generator=self._rng)
                - 1.0
            ) * torque_range[1]
        self.entity.write_external_wrench_to_sim(
            force, torque, env_ids=ids, body_ids=[self.base_body_id]
        )
        self.disturbance_wrench_timer[ids] = duration[0] + (
            duration[1] - duration[0]
        ) * torch.rand(len(ids), device=self.device, generator=self._rng)
        self.external_wrenches[ids, :3] = force[:, 0]
        self.external_wrenches[ids, 3:] = torque[:, 0]

    def update_push(self, dt: float) -> None:
        active = self.disturbance_wrench_timer > 0.0
        self.disturbance_wrench_timer.sub_(dt).clamp_(min=0.0)
        expired = active & (self.disturbance_wrench_timer <= 0.0)
        if expired.any():
            ids = expired.nonzero(as_tuple=False).flatten()
            zeros = torch.zeros(len(ids), 1, 3, device=self.device)
            self.entity.write_external_wrench_to_sim(
                zeros, zeros, env_ids=ids, body_ids=[self.base_body_id]
            )
            self.external_wrenches[ids] = 0.0
        if "robot_dr_ramp" in self.task_config.env_config["domain_rand"]:
            recovery = self.recovery_reset_type > 0
            exited = self.push_was_recovery & ~recovery
            self.push_grace_timer.sub_(dt).clamp_(min=0.0)
            self.push_grace_timer[recovery | exited] = float(
                self.task_config.env_config["domain_rand"]["push_recovery_grace_s"]
            )
            self.push_was_recovery.copy_(recovery)
        return


def actor_observation(env, stage: int = 1) -> torch.Tensor:
    runtime = getattr(env, "task_runtime", None)
    if runtime is not None:
        return runtime.actor_observation(env)
    from .config import load_task_config

    recipe = load_task_config()
    data = env.scene["robot"].data
    return build_actor_observation(
        base_ang_vel=data.root_link_ang_vel_b,
        projected_gravity=data.projected_gravity_b,
        commands=env.command_manager.get_command("twist"),
        joint_pos=data.joint_pos,
        default_joint_pos=data.default_joint_pos,
        joint_vel=data.joint_vel,
        actions=env.action_manager.action,
        recipe=SimpleNamespace(env=recipe.env_config),
    )


def critic_observation(env, stage: int = 1, width: int = 1510) -> torch.Tensor:
    runtime = getattr(env, "task_runtime", None)
    if runtime is not None:
        return runtime.critic_observation(env)
    del stage
    return torch.zeros(env.num_envs, int(width), dtype=torch.float32, device=env.device)


def task_reward(env) -> torch.Tensor:
    return env.task_runtime.compute_reward(env)


def task_termination(env) -> torch.Tensor:
    return env.task_runtime.task_termination(env)


def robot_reset_event(env, env_ids: torch.Tensor | None) -> None:
    runtime = getattr(env, "task_runtime", None)
    if runtime is not None:
        runtime.reset(env_ids)


def robot_structure_startup(env, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    runtime = getattr(env, "task_runtime", None)
    if runtime is not None:
        runtime.initialize_structure()


def robot_push_event(env, env_ids: torch.Tensor | None) -> None:
    runtime = getattr(env, "task_runtime", None)
    if runtime is not None:
        runtime.push(env_ids)


def robot_push_step(env, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    runtime = getattr(env, "task_runtime", None)
    if runtime is not None:
        runtime.update_push(float(env.step_dt))


from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

robot_structure_startup = requires_model_fields(
    "body_mass", "body_inertia", "body_ipos", recompute=RecomputeLevel.set_const
)(robot_structure_startup)
robot_reset_event = requires_model_fields(
    "geom_friction",
    "geom_solref",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    recompute=RecomputeLevel.none,
)(robot_reset_event)
