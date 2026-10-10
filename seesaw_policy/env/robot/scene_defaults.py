"""Native robot scene defaults, overridden by the seesaw task configuration."""

from __future__ import annotations
import argparse
import json
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from .asset import get_go2_entity_cfg
from .amp_obs import amp_observation
from .smoke_command import UniformVelocityCommandCfg

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


def actor_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"].data
    command = env.command_manager.get_command("twist")
    obs = torch.cat(
        (
            robot.root_link_ang_vel_b * 0.25,
            robot.projected_gravity_b,
            command * command.new_tensor((2.0, 2.0, 0.25)),
            robot.joint_pos - robot.default_joint_pos,
            robot.joint_vel * 0.05,
            env.action_manager.action,
        ),
        dim=-1,
    )
    if obs.shape != (env.num_envs, 45):
        raise AssertionError(f"Go2 actor observation shape changed: {tuple(obs.shape)}")
    return obs


def tracking_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    actual = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    command = env.command_manager.get_command("twist")[:, :2]
    return torch.exp(-torch.sum((actual - command) ** 2, dim=1) / 0.25)


def make_scene_defaults(num_envs: int) -> ManagerBasedRlEnvCfg:
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=2.0,
            terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=2.0),
            entities={"robot": get_go2_entity_cfg()},
            sensors=(
                ContactSensorCfg(
                    name="feet_ground_contact",
                    primary=ContactMatch(
                        mode="geom", pattern=("FL", "FR", "RL", "RR"), entity="robot"
                    ),
                    secondary=ContactMatch(mode="body", pattern="terrain"),
                    fields=("found", "force"),
                    reduce="netforce",
                    num_slots=1,
                    track_air_time=True,
                ),
                ContactSensorCfg(
                    name="body_ground_contact",
                    primary=ContactMatch(
                        mode="body", pattern=BODY_CONTACT_NAMES, entity="robot"
                    ),
                    secondary=ContactMatch(mode="body", pattern="terrain"),
                    fields=("found", "force"),
                    reduce="netforce",
                    num_slots=1,
                ),
            ),
        ),
        sim=SimulationCfg(
            nconmax=128,
            njmax=512,
            mujoco=MujocoCfg(
                timestep=0.005, iterations=20, cone="elliptic", impratio=100.0
            ),
        ),
        decimation=4,
        episode_length_s=5.0,
        observations={
            "actor": ObservationGroupCfg(
                terms={"proprio": ObservationTermCfg(func=actor_obs)}
            )
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.25,
                use_default_offset=True,
            )
        },
        commands={
            "twist": UniformVelocityCommandCfg(
                entity_name="robot",
                resampling_time_range=(5.0, 5.0),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
                ),
            )
        },
        rewards={"tracking_xy": RewardTermCfg(func=tracking_reward, weight=1.0)},
        terminations={
            "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(
                func=env_mdp.bad_orientation, params={"limit_angle": 1.22}
            ),
        },
    )
