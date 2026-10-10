"""Native state adapter, reusing Gym's simulator-independent boarding state."""

from copy import deepcopy
from types import SimpleNamespace
import math

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from seesaw_policy.env.equations.rewards import RobotRewardModel, RewardBatch
from seesaw_policy.env.robot.robot_runtime import (
    RobotRuntime,
    _event_env_ids,
    _quat_to_rpy_wxyz,
)
from seesaw_policy.env.boarding.task import BoardingTask
from seesaw_policy.env.boarding.curriculum import SeesawCurriculum
from seesaw_policy.env.boarding.geometry import rotate_y
from seesaw_policy.env.boarding.parameters import (
    HEIGHTS,
    WIDTHS,
    ANGLES_DEG,
    DAMPING,
    lookup,
    sample_commands,
)
from .asset import ARENA_SPACING
from .reward_scales import AUX_NAMES, TASK_SCALES
from .rewards import balance_terms, edge_risk, settling_terms, four_foot_contact_term


class SeesawRuntime(RobotRuntime):
    def __init__(self, env, recipe, **kwargs):
        # Use the robot reset and randomization path in the same physics world.
        super().__init__(env, recipe, "seesaw", **kwargs)
        self.pool = "seesaw"
        self.config = recipe.env_config["seesaw"]
        self._recovery_pool.zero_()
        self._recovery_pool_initialized = True
        self.board_curriculum = SeesawCurriculum(self.device, self.config)
        self.boarding = BoardingTask(self.num_envs, self.device, self.config, self.dt)
        self.stable_hold_seconds = torch.zeros(self.num_envs, device=self.device)
        self.board_contact_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.entry_target_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.board_height = lookup(HEIGHTS, self.levels)
        self.board_half_width = lookup(WIDTHS, self.levels) / 2
        self.board_half_length = torch.full(
            (self.num_envs,), self.config["half_length"], device=self.device
        )
        self.board_max_angle = lookup(ANGLES_DEG, self.levels) * (math.pi / 180.0)
        self.board_mass = torch.full((self.num_envs,), 12.0, device=self.device)
        self.board_damping = lookup(DAMPING, self.levels)
        self.board_friction = torch.ones(self.num_envs, device=self.device)
        from .domain_randomization import BoardDomainRandomization

        self.board_randomization = BoardDomainRandomization(self)
        from .perturbations import SeesawPerturbations

        perturbation = self.config.get("perturbations", {})
        self.perturbations = (
            SeesawPerturbations(self, perturbation)
            if perturbation.get("enabled", False)
            else None
        )
        self.play_levels = None
        self.last_success_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.completed_episodes = 0
        self.completed_mounts = 0
        self.completed_successes = 0
        self.completed_deadlines = 0
        env_cfg = deepcopy(recipe.env_config)
        rewards = env_cfg["rewards"]
        rewards["scales"] = {
            k: v for k, v in rewards["scales"].items() if k in AUX_NAMES
        }
        rewards["scales"]["correct_base_height"] = -7.5
        rewards["weights"]["correct_base_height"] = 0.4
        rewards["only_positive_rewards"] = False
        self.reward_model = RobotRewardModel(
            SimpleNamespace(
                env=env_cfg,
                train=recipe.train_config,
                control_dt=self.dt,
                stage=self.stage,
            ),
            decomposition_rtol=1e-6,
        )

    @property
    def board_angle(self):
        positions = torch.stack(
            [self.env.scene[f"seesaw_l{i}"].data.joint_pos[:, 0] for i in range(10)],
            dim=1,
        )
        return positions.gather(1, self.levels[:, None]).squeeze(1)

    @property
    def board_rate(self):
        velocities = torch.stack(
            [self.env.scene[f"seesaw_l{i}"].data.joint_vel[:, 0] for i in range(10)],
            dim=1,
        )
        return velocities.gather(1, self.levels[:, None]).squeeze(1)

    def board_coordinates(self, points):
        center = self.env.scene.env_origins.clone()
        center[:, 2] += self.board_height
        if points.ndim == 3:
            center = center[:, None]
        return rotate_y(points - center, -self.board_angle)

    def reset(self, env_ids=None):
        ids = _event_env_ids(self.env, env_ids)
        self.stable_hold_seconds[ids] = 0.0
        self.board_contact_count[ids] = 0
        if self.perturbations is not None:
            self.perturbations.reset(ids)
        valid = ids[self.env.episode_length_buf[ids] > 0]
        if valid.numel():
            b = self.boarding
            self.board_curriculum.record(
                self.levels[valid],
                (b.dynamic_success[valid] & ~b.failure[valid]).float(),
                b.mounted[valid].float(),
                b.response_time[valid],
                b.response_good[valid],
            )
            self.completed_episodes += len(valid)
            self.completed_mounts += int(b.mounted[valid].sum())
            self.completed_successes += int(
                (b.dynamic_success[valid] & ~b.failure[valid]).sum()
            )
            self.completed_deadlines += int(b.boarding_timeout[valid].sum())
        ceiling = self.board_curriculum.level
        levels = torch.full_like(ids, ceiling)
        if ceiling:
            replay = (
                torch.rand(len(ids), device=self.device)
                < self.config["replay_fraction"]
            )
            levels = torch.where(
                replay, torch.randint(ceiling, (len(ids),), device=self.device), levels
            )
        if self.play_levels is not None:
            levels = self.play_levels[ids]
        self.levels[ids] = levels
        self.board_height[ids] = lookup(HEIGHTS, levels)
        self.board_half_width[ids] = lookup(WIDTHS, levels) / 2
        self.board_half_length[ids] = self.config["half_length"]
        self.board_max_angle[ids] = lookup(ANGLES_DEG, levels) * (math.pi / 180.0)
        self.board_mass[ids] = 12.0
        self.board_damping[ids] = lookup(DAMPING, levels)
        self.board_friction[ids] = 1.0
        origins = self.env.scene.env_origins
        origins[ids] = 0.0
        origins[ids, 0] = levels.remainder(5) * ARENA_SPACING[0]
        origins[ids, 1] = levels.div(5, rounding_mode="floor") * ARENA_SPACING[1]
        origins[ids] += origins.new_tensor(
            self.config.get("arena_offset", (0.0, 0.0, 0.0))
        )
        # Shared native reset preserves the AMP pose and recovery-bank state.
        super().reset(ids)
        self.board_randomization.apply(ids)
        side = torch.where(torch.rand(len(ids), device=self.device) < 0.5, 1.0, -1.0)
        angle = side * self.board_max_angle[ids]
        for level in range(10):
            selected = ids[levels == level]
            if selected.numel():
                values = angle[levels == level, None]
                self.env.scene[f"seesaw_l{level}"].write_joint_state_to_sim(
                    values, torch.zeros_like(values), env_ids=selected
                )
        # Shared reset has written qpos/qvel, but forward() has not run yet.
        # xpos/xquat/cvel still describe the previous episode and must not be
        # copied back into the fresh AMP/recovery reset state.
        index = self.entity.indexing
        data = self.env.sim.data
        pose = data.qpos[ids[:, None], index.free_joint_q_adr[None, :]].clone()
        velocity = data.qvel[ids[:, None], index.free_joint_v_adr[None, :]].clone()
        root = torch.cat(
            (pose, velocity[:, :3], quat_apply(pose[:, 3:7], velocity[:, 3:])), dim=1
        )
        # v12 level varies tilt only; keep spawn difficulty independent.
        spawn_level = (
            levels if self.config["version"] == 11 else torch.zeros_like(levels)
        )
        gap = 0.45 + torch.rand(len(ids), device=self.device) * (
            0.15 + 0.50 * spawn_level / 9.0
        )
        root[:, 0] = origins[ids, 0] + side * (
            self.board_half_length[ids] * angle.cos() + gap
        )
        root[:, 1] = origins[ids, 1] + (
            2 * torch.rand(len(ids), device=self.device) - 1
        ) * self.board_half_width[ids] * (0.125 + 0.475 * spawn_level / 9.0)
        target = torch.zeros_like(root[:, :3])
        target[:, 0] = side * self.board_half_length[ids] * (1.4 / 1.8)
        target[:, 2] = 0.35
        self.entry_target_local[ids] = target
        entry = rotate_y(target, angle) + origins[ids]
        entry[:, 2] += self.board_height[ids]
        desired_yaw = torch.atan2(entry[:, 1] - root[:, 1], entry[:, 0] - root[:, 0])
        old_yaw = _quat_to_rpy_wxyz(root[:, 3:7])[:, 2]
        turn = root.new_zeros((len(ids), 4))
        turn[:, 0] = ((desired_yaw - old_yaw) / 2).cos()
        turn[:, 3] = ((desired_yaw - old_yaw) / 2).sin()
        root[:, 3:7] = quat_mul(turn, root[:, 3:7])
        root[:, 7:10] = quat_apply(turn, root[:, 7:10])
        root[:, 10:13] = quat_apply(turn, root[:, 10:13])
        # AMP/recovery poses differ from the default stance. Fit their actual
        # collision bounds above ground and outside the lowered board end.
        from seesaw_policy.env.robot.spawn_geometry import Go2SpawnClearance

        if not hasattr(self, "_spawn_clearance"):
            self._spawn_clearance = Go2SpawnClearance(self.joint_names, self.device)
        joints = data.qpos[ids[:, None], index.joint_q_adr[None, :]]
        low, high = self._spawn_clearance.collision_bounds(joints, root[:, 3:7])
        root[:, 2] = torch.maximum(root[:, 2], origins[ids, 2] + 0.015 - low[:, 2])
        board_extent = (
            self.board_half_length[ids] * angle.cos() + 0.02 * angle.sin().abs()
        )
        near_extent = torch.where(side > 0, -low[:, 0], high[:, 0])
        distance_x = side * (root[:, 0] - origins[ids, 0])
        root[:, 0] = origins[ids, 0] + side * torch.maximum(
            distance_x, board_extent + near_extent + 0.05
        )
        self.entity.write_root_state_to_sim(root, env_ids=ids)
        center = origins[ids].clone()
        center[:, 2] += self.board_height[ids]
        distance = (rotate_y(root[:, :3] - center, -angle) - target).norm(dim=-1)
        self.boarding.reset(ids, distance)

    def push(self, env_ids):
        # The new mounted pulse scheduler replaces the seesaw pool's native
        # four-second wrench event; flat/rough retain their Stage3 push path.
        if self.perturbations is None:
            super().push(env_ids)

    def update_push(self, dt):
        super().update_push(dt)
        if self.perturbations is not None:
            self.perturbations.step(dt)

    def sample_commands(self, ids, term):
        commands = sample_commands(self.levels[ids], self.boarding.mounted[ids])
        self.recovery_target_command[ids] = commands
        term.command[ids] = torch.where(
            self.recovery_reset_type[ids, None] > 0, 0.0, commands
        )

    def _critic_action_delay(self):
        decimation = round(self.dt / self.physics_dt)
        current = self.action_delay_steps.float() / decimation
        current = torch.where(self.env.episode_length_buf == 0, 0.0, current)
        maximum = (
            self.action_delay_max_steps.amax().clamp(0, decimation).float() / decimation
        )
        return torch.stack((current, maximum.expand_as(current)), dim=1)

    def task_termination(self, env=None, *, measurements=None):
        common = super().task_termination(env, measurements=measurements)
        state = self.entity.data
        b, c = self.boarding, self.config
        active = ~self.recovery_reward_active
        if self.active_mask is not None:
            active = active & self.active_mask
        local = self.board_coordinates(state.root_link_pos_w)
        position = state.root_link_pos_w - self.env.scene.env_origins
        tilt = torch.acos((-state.projected_gravity_b[:, 2]).clamp(-1, 1))
        failure = common | (
            active
            & (
                (position[:, 0].abs() > c["arena_half_x"])
                | (position[:, 1].abs() > c["arena_half_y"])
                | (position[:, 2] < 0.12)
                | (tilt > c["failure_tilt"])
            )
        )
        failure |= ~torch.isfinite(self.board_angle) | ~torch.isfinite(self.board_rate)
        feet = state.body_link_pos_w[:, self.feet_ids]
        f = self.board_coordinates(feet)
        contact = -self.env.scene["feet_ground_contact"].data.force[..., 2] > 1.0
        on_board = (
            (f[..., 0].abs() < self.board_half_length[:, None])
            & (f[..., 1].abs() < self.board_half_width[:, None])
            & (f[..., 2] >= 0.005)
            & (f[..., 2] < 0.08)
            & contact
        )
        on_ground = contact & ~on_board & (feet[..., 2] < 0.08)
        self.board_contact_count.copy_(on_board.sum(-1))
        inside = (
            (local[:, 0].abs() < self.board_half_length - 0.05)
            & (local[:, 1].abs() < self.board_half_width * 0.9)
            & (local[:, 2] > 0.14)
        )
        outside = (local[:, 0].abs() > self.board_half_length + 0.25) | (
            local[:, 1].abs() > self.board_half_width + 0.08
        )
        b.update(
            (local - self.entry_target_local).norm(dim=-1),
            local[:, :2].norm(dim=-1),
            self.board_contact_count,
            on_ground.sum(-1),
            inside,
            local[:, 2] < 0.14,
            outside,
            self.board_angle,
            self.board_rate,
            tilt,
            failure,
            enabled=active,
        )
        success = active & b.dynamic_success & ~b.failure
        self.last_success_event.copy_(success)
        self.episode_success.copy_(success)
        self.episode_failure |= b.failure
        done = common | b.failure | success
        self.last_task_termination.copy_(done)
        self.last_fall_event |= b.failure
        return done

    def tracking_velocity(self):
        state = self.entity.data
        yaw = _quat_to_rpy_wxyz(state.root_link_quat_w)[:, 2]
        heading = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
        angle = self.board_angle * self.boarding.mounted
        forward = rotate_y(heading, -angle)
        forward[:, 2] = 0.0
        forward = torch.nn.functional.normalize(forward, dim=-1)
        lateral = torch.stack(
            (-forward[:, 1], forward[:, 0], torch.zeros_like(yaw)), -1
        )
        r = state.root_link_pos_w - self.env.scene.env_origins
        r = r.clone()
        r[:, 2] -= self.board_height
        rate = self.board_rate * self.boarding.mounted
        surface = torch.stack(
            (rate * r[:, 2], torch.zeros_like(rate), -rate * r[:, 0]), -1
        )
        relative = rotate_y(state.root_link_lin_vel_w - surface, -angle)
        return torch.stack(
            ((relative * forward).sum(-1), (relative * lateral).sum(-1)), -1
        )

    def compute_reward(self, env=None):
        super().compute_reward(env)
        b = self.boarding
        local = self.board_coordinates(self.entity.data.root_link_pos_w)
        velocity = self.tracking_velocity()
        safe = torch.stack(
            (self.board_half_length * (1.25 / 1.8), self.board_half_width * 0.65), -1
        )
        raw = dict(
            seesaw_progress=b.progress,
            seesaw_mount=b.mount_event.float(),
            seesaw_angle_progress=b.angle_progress,
            seesaw_failure=b.failure.float(),
            seesaw_success=b.dynamic_success.float(),
            seesaw_four_foot=four_foot_contact_term(b, self.board_contact_count),
            seesaw_level=b.balance_reward(local[:, :2].norm(dim=-1), self.board_angle),
            seesaw_rate=self.board_rate.square() * b.supported,
            seesaw_edge=torch.relu(local[:, :2].abs() - safe).square().sum(-1)
            * b.mounted,
            seesaw_tracking=torch.exp(
                -(velocity - self.command[:, :2]).square().sum(-1) / 0.6**2
            )
            * (~b.failure),
            seesaw_speed_excess=torch.relu(
                velocity.norm(dim=-1) - self.config["mounted_speed_limit"]
            ).square()
            * b.mounted
            * (~b.failure),
        )
        scales = TASK_SCALES
        if self.config.get("reward_version", 1) >= 2:
            raw.update(
                balance_terms(
                    b,
                    local,
                    self.board_angle,
                    self.board_rate,
                    self.config,
                    max_angle=self.board_max_angle,
                )
            )
            feet_local = self.board_coordinates(
                self.entity.data.body_link_pos_w[:, self.feet_ids]
            )
            raw["seesaw_edge"] = (
                edge_risk(
                    local,
                    feet_local,
                    self.board_half_width,
                    self.config,
                    half_length=self.board_half_length,
                )
                * b.mounted
                * (~b.failure)
            )
            scales = dict(TASK_SCALES, **self.config["reward_scales"])
            if self.config["reward_version"] >= 3:
                raw.update(
                    settling_terms(
                        b,
                        self.board_angle,
                        self.board_rate,
                        self.stable_hold_seconds,
                        self.config,
                        self.dt,
                        max_angle=self.board_max_angle,
                    )
                )
        events = {
            "seesaw_progress",
            "seesaw_angle_progress",
            "seesaw_mount",
            "seesaw_failure",
            "seesaw_success",
        }
        enabled = ~self.recovery_reward_active & ~self.last_velocity_nonfinite_event
        terms = {
            name: value * scales[name] * (1.0 if name in events else self.dt) * enabled
            for name, value in raw.items()
        }
        batch = self.last_reward_batch
        task = batch.task + sum(terms.values())
        self.last_raw_reward_terms.update(raw)
        self.last_final_reward_terms.update(terms)
        self.last_tracking_reward.copy_(terms["seesaw_tracking"])
        self.last_reward_batch = RewardBatch(
            task + batch.auxiliary,
            task,
            batch.auxiliary,
            self.last_final_reward_terms,
            self.last_raw_reward_terms,
        )
        # Preserve the command that produced this transition's reward. Publish
        # zero board commands for the next action once mount is confirmed.
        mounted = (
            b.mounted if self.active_mask is None else b.mounted & self.active_mask
        )
        self.command[mounted] = 0.0
        self.recovery_target_command[mounted] = 0.0
        return self.last_reward_batch.total

    def amp_observation(self):
        observation = super().amp_observation()
        if not self.task_config.env_config["env"].get("amp_mask_base_height", False):
            clearance = (
                self.board_coordinates(self.entity.data.root_link_pos_w)[:, 2] - 0.02
            )
            fraction = self.boarding.support_fraction
            observation[:, -1] = (
                fraction * clearance
                + (1 - fraction) * self.entity.data.root_link_pos_w[:, 2]
            )
        return observation
