from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, replace
import math
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from .asset import TRUNK_CONTACT_GEOM, get_go2_entity_cfg
from .scene_defaults import make_scene_defaults
from .terrain_rays import make_terrain_ray_sensors
from .robot_runtime import (
    actor_observation,
    critic_observation,
    robot_push_event,
    robot_push_step,
    robot_reset_event,
    task_reward,
    robot_structure_startup,
    task_termination,
)


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
    entity_name: str
    stage: int
    pool: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    yaw_range: tuple[float, float]
    zero_probability: float = 1.0 / 7.0

    def build(self, env) -> "VelocityCommand":
        return VelocityCommand(self, env)


class VelocityCommand(CommandTerm):

    def __init__(self, cfg: VelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self) -> None:
        pass

    @staticmethod
    def _truncated_normal(
        mean: float, std: float, lo: float, hi: float, count: int, device
    ) -> torch.Tensor:
        root2 = math.sqrt(2.0)
        cdf = lambda x: 0.5 * (
            1.0 + torch.erf(torch.as_tensor(x, device=device) / root2)
        )
        low, high = (cdf((lo - mean) / std), cdf((hi - mean) / std))
        p = low + (high - low) * torch.rand(count, device=device)
        return (
            mean + std * root2 * torch.erfinv(2.0 * p.clamp(1e-06, 1.0 - 1e-06) - 1.0)
        ).clamp(lo, hi)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        runtime = getattr(self._env, "task_runtime", None)
        if runtime is not None and runtime.pool == "seesaw":
            runtime.sample_commands(env_ids, self)
            return
        recovery = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if runtime is not None:
            recovery = runtime.recovery_reset_type[env_ids] > 0
        x = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.x_range)
        y = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.y_range)
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.yaw_range
        )
        zero = torch.rand(len(env_ids), device=self.device) < self.cfg.zero_probability
        x[zero] = 0.0
        y[zero] = 0.0
        yaw[zero] = 0.0
        sampled = torch.stack((x, y, yaw), dim=1)
        if runtime is not None:
            runtime.recovery_target_command[env_ids] = sampled
        self.vel_command_b[env_ids] = torch.where(recovery[:, None], 0.0, sampled)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    max_delay_substeps: int = 4
    stage: int = 1

    def build(self, env) -> "DelayedJointPositionAction":
        return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
    """Position-PD action with a per-world 0--4 physics-substep delay."""

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self.max_delay_substeps = int(cfg.max_delay_substeps)
        self.action_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motor_zero_offsets = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._previous_raw_action = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._substep_counter = 0
        self._executed_target = self._entity.data.default_joint_pos[
            :, self._target_ids
        ].clone()
        self.action_fifo = torch.zeros(self.num_envs, 84, device=self.device)

    @property
    def executed_target(self) -> torch.Tensor:
        return self._executed_target

    def process_actions(self, actions: torch.Tensor) -> None:
        self._previous_raw_action.copy_(self._raw_actions)
        self._raw_actions[:] = actions
        runtime = self._env.task_runtime
        limit = runtime.action_delay_max_steps
        sampled = (torch.rand(self.num_envs, device=self.device) * (limit + 1)).long()
        self.action_delay_steps.copy_(sampled)
        runtime.action_delay_steps.copy_(sampled)
        self.action_fifo[:, :-12] = self.action_fifo[:, 12:].clone()
        self.action_fifo[:, -12:] = actions
        self._substep_counter = 0

    def apply_actions(self) -> None:
        delayed = self.action_delay_steps > self._substep_counter
        raw = torch.where(
            delayed[:, None], self._previous_raw_action, self._raw_actions
        )
        default = self._entity.data.default_joint_pos[:, self._target_ids]
        target = default + raw * float(self.cfg.scale) + self.motor_zero_offsets
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = target - encoder_bias
        self._executed_target.copy_(target)
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
        self._substep_counter += 1

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._previous_raw_action[env_ids] = 0.0
        self.action_fifo[env_ids] = 0.0
        self._substep_counter = 0


def _command_cfg(recipe: TaskConfig, pool: str) -> VelocityCommandCfg:
    curriculum = recipe.env_config.get("staged_curriculum", {})
    max_speed = float(curriculum.get("linear_speed_max", 1.5))
    lateral = float(curriculum.get("lateral_speed_max", 0.5))
    max_yaw = float(curriculum.get("yaw_speed_max", 1.0))
    return VelocityCommandCfg(
        entity_name="robot",
        stage=3,
        pool=pool,
        resampling_time_range=(5.0, 5.0),
        x_range=(-max_speed, max_speed),
        y_range=(-lateral, lateral),
        yaw_range=(-max_yaw, max_yaw),
        zero_probability=float(curriculum.get("velocity_zero_probability", 1.0 / 7.0)),
    )


def make_task_env_cfg(
    recipe: TaskConfig, pool: str, terrain_cfg, num_envs: int
) -> ManagerBasedRlEnvCfg:
    """Create one native manager-based MJLab task pool from a frozen recipe."""
    if num_envs < 1:
        raise ValueError("A task pool must contain at least one environment")
    cfg = make_scene_defaults(num_envs)
    sensors = list(cfg.scene.sensors)
    body_sensor_ids = [
        index
        for index, sensor in enumerate(sensors)
        if getattr(sensor, "name", None) == "body_ground_contact"
    ]
    if len(body_sensor_ids) != 1:
        raise RuntimeError(
            "Expected exactly one body_ground_contact sensor in the base MJLab config"
        )
    body_sensor_index = body_sensor_ids[0]
    sensors[body_sensor_index] = replace(
        sensors[body_sensor_index],
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
    )
    sensors.append(
        replace(
            sensors[body_sensor_index],
            name="trunk_ground_contact",
            primary=ContactMatch(
                mode="geom", pattern=TRUNK_CONTACT_GEOM, entity="robot"
            ),
        )
    )
    cfg.scene = SceneCfg(
        num_envs=num_envs,
        env_spacing=8.0,
        terrain=deepcopy(terrain_cfg),
        entities={"robot": get_go2_entity_cfg()},
        sensors=tuple(sensors) + make_terrain_ray_sensors(recipe),
    )
    cfg.scene.terrain.num_envs = num_envs
    cfg.sim = SimulationCfg(
        nconmax=128,
        njmax=512,
        mujoco=MujocoCfg(
            timestep=float(recipe.env_config["sim"]["dt"]),
            iterations=20,
            cone="elliptic",
            impratio=100.0,
        ),
    )
    cfg.decimation = int(recipe.env_config["control"]["decimation"])
    cfg.episode_length_s = float(recipe.env_config["env"].get("episode_length_s", 20.0))
    cfg.scale_rewards_by_dt = False
    cfg.auto_reset = True
    cfg.actions = {
        "joint_pos": DelayedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=float(recipe.env_config["control"]["action_scale"]),
            use_default_offset=True,
            max_delay_substeps=int(
                recipe.env_config.get("domain_rand", {}).get(
                    "action_delay_max_steps_end", 4
                )
            ),
            stage=3,
        )
    }
    cfg.commands = {"twist": _command_cfg(recipe, pool)}
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={
                "proprio": ObservationTermCfg(
                    func=actor_observation, params={"stage": 3}
                )
            }
        ),
        "critic": ObservationGroupCfg(
            terms={
                "shared_privileged": ObservationTermCfg(
                    func=critic_observation,
                    params={
                        "stage": 3,
                        "width": int(recipe.env_config["env"]["num_privileged_obs"]),
                    },
                )
            }
        ),
    }
    cfg.rewards = {"geometry": RewardTermCfg(func=task_reward, weight=1.0)}
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
        "geometry_task": TerminationTermCfg(func=task_termination),
    }
    dr = recipe.env_config.get("domain_rand", {})
    push_enabled = bool(
        dr.get("push_robots", False) or dr.get("disturbance_flat_wrench", False)
    )
    cfg.events = {
        "geometry_structure": EventTermCfg(
            func=robot_structure_startup, mode="startup"
        ),
        "geometry_reset": EventTermCfg(func=robot_reset_event, mode="reset"),
    }
    if push_enabled:
        push_interval = float(dr.get("push_interval_s", 4.0))
        cfg.events["geometry_push"] = EventTermCfg(
            func=robot_push_event,
            mode="interval",
            interval_range_s=(push_interval, push_interval),
        )
        cfg.events["geometry_push_decay"] = EventTermCfg(
            func=robot_push_step, mode="step"
        )
    return cfg
