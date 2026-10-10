"""Single-task vector interface; every world trains on a seesaw."""

from __future__ import annotations
from dataclasses import replace
import math
import json
import time
from types import SimpleNamespace
from typing import Any
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg
from .robot.config_view import ConfigNode
from .robot.scene import make_task_env_cfg
from .board.asset import board_entities
from .board.runtime import SeesawRuntime
from .board.domain_randomization import seesaw_reset_event


def _cat_tree(items: list[Any]) -> Any:
    first = items[0]
    if torch.is_tensor(first):
        return torch.cat(items, dim=0)
    if isinstance(first, dict):
        return {key: _cat_tree([item[key] for item in items]) for key in first}
    if isinstance(first, tuple):
        return tuple(
            _cat_tree([item[index] for item in items]) for index in range(len(first))
        )
    return first


class SeesawEnv:
    shared_privileged = True

    def __init__(
        self,
        num_envs,
        device="cuda:0",
        recipe=None,
        *,
        global_num_envs=None,
        rank=0,
        world_size=1,
        seed=0,
        playback=False,
    ):
        from seesaw_policy.task import make_task_recipe

        self.task_config = recipe or make_task_recipe()
        self.stage = 3
        self.playback = playback
        self.freeze_curriculum = playback
        self.num_envs = int(num_envs)
        self.rank, self.world_size = int(rank), int(world_size)
        self.global_num_envs = int(global_num_envs or num_envs * world_size)
        self.seed = int(seed)
        expected = self.global_num_envs // world_size + int(
            rank < self.global_num_envs % world_size
        )
        if (
            world_size < 1
            or not 0 <= rank < world_size
            or self.num_envs != expected
            or self.num_envs < 1
        ):
            raise ValueError("Invalid distributed seesaw environment allocation")
        self.device = torch.device(device)
        self.sim_device = str(device)
        self.dt = self.task_config.control_dt
        self.max_episode_length = math.ceil(
            self.task_config.env_config["env"].get("episode_length_s", 20.0) / self.dt
        )
        self.num_obs, self.num_actions = 45, 12
        self.num_privileged_obs = int(
            self.task_config.env_config["env"]["num_privileged_obs"]
        )
        self.amp_include_base_lin_vel = bool(
            self.task_config.env_config["env"].get("amp_include_base_lin_vel", True)
        )
        self.critic_extra_layout = dict(
            self.task_config.train_config["policy"]["critic_privileged_layout"]
        )
        self.terrain_schema = self.task_config.train_config["policy"][
            "critic_terrain_schema"
        ]
        self.cfg = ConfigNode.from_value(self.task_config.env_config)
        self.cfg.env.num_envs = self.num_envs
        self.cfg.env.test = False
        self.cfg.env.critic_extra_privileged = True
        self.cfg.env.critic_terrain_embedding_dim = int(
            self.task_config.train_config["policy"]["critic_terrain_embedding_dim"]
        )
        self.cfg.geometry = ConfigNode.from_value(
            self.task_config.env_config["geometry"]
        )
        self.cfg.depth = ConfigNode.from_value(
            self.task_config.env_config.get("depth", {})
        )
        self.staged_curriculum = None
        self.collect_reward_details = False
        self.terrain_command_levels = torch.zeros(34, device=self.device)
        self.active_terrain_sample_ids = torch.tensor([10], device=self.device)
        self._profile_step_limit = self._profile_step_count = 0
        self.lab_envs = []
        self.runtimes = []
        self.pool_names = []
        self.pool_sizes = []
        self._recent_curriculum_events = []
        self._build_task_pools(device)
        self.terrain_data = SimpleNamespace(schema=self.terrain_schema, data=None)
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.tracking_reward_buf = torch.zeros(self.num_envs, device=self.device)
        self.last_raw_reward_terms = {}
        self.last_final_reward_terms = {}
        self._obs = torch.zeros(self.num_envs, 45, device=self.device)
        self._privileged = torch.zeros(
            self.num_envs, self.num_privileged_obs, device=self.device
        )
        self._refresh_from_manager_obs()
        self._refresh_pool_metadata()
        for name in (
            "last_fall_buf",
            "last_base_contact_fall_buf",
            "last_orientation_guard_reset_buf",
            "last_velocity_guard_reset_buf",
            "last_velocity_nonfinite_reset_buf",
            "last_velocity_lin_guard_reset_buf",
            "last_velocity_ang_guard_reset_buf",
            "last_recovery_timeout_buf",
        ):
            setattr(self, name, torch.zeros_like(self.reset_buf))

    def _build_task_pools(self, device):
        self.global_pool_sizes = {"seesaw": self.global_num_envs}
        terrain = TerrainEntityCfg(
            terrain_type="plane", env_spacing=8.0, num_envs=self.num_envs
        )
        cfg = make_task_env_cfg(self.task_config, "seesaw", terrain, self.num_envs)
        cfg.sim.mujoco = replace(cfg.sim.mujoco, timestep=cfg.sim.mujoco.timestep / 2)
        cfg.decimation *= 2
        randomized = self.task_config.env_config["seesaw"]["domain_rand"]["enabled"]
        cfg.scene.entities.update(board_entities(randomized))
        cfg.events["geometry_reset"] = replace(
            cfg.events["geometry_reset"], func=seesaw_reset_event
        )
        cfg.scene.sensors = tuple(
            (
                replace(sensor, secondary=None)
                if isinstance(sensor, ContactSensorCfg)
                else sensor
            )
            for sensor in cfg.scene.sensors
        )
        cfg.commands["twist"].resampling_time_range = (2.0, 2.0)
        lab = ManagerBasedRlEnv(cfg, device=device)
        lab.scene.env_origins.zero_()
        runtime = SeesawRuntime(
            lab, self.task_config, seed=self.seed + 100003 * self.rank
        )
        lab.task_runtime = runtime
        if "startup" in lab.event_manager.available_modes:
            lab.event_manager.apply(mode="startup")
            lab.sim.forward()
        lab.reset()
        self.lab_envs.append(lab)
        self.runtimes.append(runtime)
        self.pool_names.append("seesaw")
        self.pool_sizes.append(self.num_envs)
        self.distributed_amp_env_ids = (
            self.rank
            + torch.arange(self.num_envs, device=self.device) * self.world_size
        )

    def _pool_slices(self):
        start = 0
        for env, size, runtime in zip(
            self.lab_envs, self.pool_sizes, self.runtimes, strict=True
        ):
            yield env, size, runtime, slice(start, start + size)
            start += size

    def _refresh_from_manager_obs(self) -> None:
        actor, critic = [], []
        for lab_env in self.lab_envs:
            actor.append(lab_env.obs_buf["actor"])
            critic.append(lab_env.obs_buf["critic"])
        self._obs = torch.cat(actor, dim=0)
        self._privileged = torch.cat(critic, dim=0)

    @property
    def last_turn_over_reward_mask(self) -> torch.Tensor:
        """Recovery rows whose reward was active in the most recent step."""
        return torch.cat([r.last_turn_over_reward_mask for r in self.runtimes])

    @staticmethod
    def _tree_cat(items: list[Any]) -> Any:
        return _cat_tree(items)

    def enable_terrain_data(self):
        # PPO packs these fields directly into its reusable geometry buffer.
        # Materializing another dense tensor here only allocates and discards it.
        data = [runtime.read_terrain(pack=False)[0] for runtime in self.runtimes]
        self.terrain_data.data = self._tree_cat(data)
        return self.terrain_data

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            for lab_env in self.lab_envs:
                lab_env.reset()
            self.reset_buf.zero_()
            self.time_out_buf.zero_()
            self.last_fall_buf.zero_()
            self.last_base_contact_fall_buf.zero_()
            self.last_orientation_guard_reset_buf.zero_()
            self.last_velocity_guard_reset_buf.zero_()
            self.last_velocity_nonfinite_reset_buf.zero_()
            self.last_velocity_lin_guard_reset_buf.zero_()
            self.last_velocity_ang_guard_reset_buf.zero_()
            self.last_recovery_timeout_buf.zero_()
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            for pool_index, (_, _, _, pool_slice) in enumerate(self._pool_slices()):
                selected = (
                    ids[(ids >= pool_slice.start) & (ids < pool_slice.stop)]
                    - pool_slice.start
                )
                if selected.numel():
                    self.lab_envs[pool_index].reset(env_ids=selected)
            self.reset_buf[ids] = False
            self.time_out_buf[ids] = False
        self._refresh_from_manager_obs()
        self._refresh_pool_metadata()
        self.enable_terrain_data()
        return self._obs, self._privileged

    def _step(self, actions: torch.Tensor):
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Expected actions {(self.num_envs, self.num_actions)}, got {tuple(actions.shape)}"
            )
        profile = self._profile_step_count < self._profile_step_limit
        if profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        profile_started = time.perf_counter() if profile else None
        actors, critics, tasks, auxiliaries, dones, timeouts, trackers = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        pool_timings = []
        raw_terms: dict[str, list[torch.Tensor]] = {}
        final_terms: dict[str, list[torch.Tensor]] = {}
        for lab_env, size, runtime, pool_slice in self._pool_slices():
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            pool_started = time.perf_counter() if profile else None
            if getattr(runtime, "optimize_reward_details", False):
                runtime.collect_reward_details = self.collect_reward_details
            obs, _, terminated, truncated, _ = lab_env.step(actions[pool_slice])
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if profile:
                pool_timings.append(
                    {
                        "pool": runtime.pool,
                        "envs": int(size),
                        "seconds": round(time.perf_counter() - pool_started, 6),
                    }
                )
            done = terminated | truncated
            # Numerical failures are terminal, including at an episode limit;
            # do not bootstrap a corrupted terminal transition as a timeout.
            truncated = truncated & ~runtime.last_velocity_nonfinite_event
            runtime.finalize_step(done)
            actors.append(obs["actor"])
            critics.append(obs["critic"])
            tasks.append(runtime.last_reward_batch.task.clone())
            auxiliaries.append(runtime.last_reward_batch.auxiliary.clone())
            dones.append(done)
            timeouts.append(truncated)
            trackers.append(runtime.last_tracking_reward.clone())
            for name, value in runtime.last_raw_reward_terms.items():
                raw_terms.setdefault(name, []).append(value.clone())
            for name, value in runtime.last_final_reward_terms.items():
                final_terms.setdefault(name, []).append(value.clone())
        self._obs = torch.cat(actors)
        self._privileged = torch.cat(critics)
        from .robot.policy_input_guard import invalid_policy_rows

        bad_input = invalid_policy_rows(self._obs, self._privileged)
        amp_input = self.get_amp_observations()
        bad_input |= ~torch.isfinite(amp_input).all(dim=1) | (
            amp_input.abs() > 1.0e4
        ).any(dim=1)
        task_reward = torch.cat(tasks)
        auxiliary_reward = torch.cat(auxiliaries)
        rewards = task_reward + auxiliary_reward
        done_buf = torch.cat(dones).bool()
        self.reset_buf.copy_(done_buf)
        self.time_out_buf.copy_(torch.cat(timeouts).bool())
        self.tracking_reward_buf.copy_(torch.cat(trackers))
        self.last_raw_reward_terms = {
            name: torch.cat(values) for name, values in raw_terms.items()
        }
        self.last_final_reward_terms = {
            name: torch.cat(values) for name, values in final_terms.items()
        }
        for target, source in (
            ("last_fall_buf", "last_fall_event"),
            ("last_base_contact_fall_buf", "last_base_contact_event"),
            ("last_orientation_guard_reset_buf", "last_orientation_event"),
            ("last_velocity_guard_reset_buf", "last_velocity_event"),
            ("last_velocity_nonfinite_reset_buf", "last_velocity_nonfinite_event"),
            ("last_velocity_lin_guard_reset_buf", "last_velocity_linear_event"),
            ("last_velocity_ang_guard_reset_buf", "last_velocity_angular_event"),
            ("last_recovery_timeout_buf", "last_recovery_timeout_event"),
        ):
            getattr(self, target).copy_(
                torch.cat([getattr(runtime, source) for runtime in self.runtimes])
            )
        self._refresh_pool_metadata()
        if profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        terrain_started = time.perf_counter() if profile else None
        self.enable_terrain_data()
        if bool(bad_input.any()):
            import os

            ids = bad_input.nonzero(as_tuple=False).flatten()
            count = getattr(self, "_policy_input_reset_count", 0) + len(ids)
            self._policy_input_reset_count = count
            if count <= 10 or count % 100 == 0:
                print(
                    f"MJLAB_POLICY_INPUT_RESET rank={os.environ.get('RANK', 'local')} "
                    f"count={count} ids={ids.tolist()}",
                    flush=True,
                )
            self.reset(ids)
            still_bad = invalid_policy_rows(self._obs, self._privileged)
            reset_amp_input = self.get_amp_observations()
            still_bad |= ~torch.isfinite(reset_amp_input).all(dim=1) | (
                reset_amp_input.abs() > 1.0e4
            ).any(dim=1)
            if bool(still_bad.any()):
                raise RuntimeError(
                    "Policy input still invalid after reset: "
                    f"{still_bad.nonzero(as_tuple=False).flatten().tolist()}"
                )
            task_reward[ids] = 0.0
            auxiliary_reward[ids] = 0.0
            rewards[ids] = 0.0
            done_buf[ids] = True
            self.reset_buf[ids] = True
            self.time_out_buf[ids] = False
            self.last_velocity_nonfinite_reset_buf[ids] = True
            self.tracking_reward_buf[ids] = 0.0
        if profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if profile:
            print(
                "MJLAB_STEP_TIMING "
                + json.dumps(
                    {
                        "rank": self.rank,
                        "step": self._profile_step_count,
                        "pools": pool_timings,
                        "wrapper_terrain_read_seconds": round(
                            time.perf_counter() - terrain_started, 6
                        ),
                        "total_seconds": round(
                            time.perf_counter() - profile_started, 6
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            self._profile_step_count += 1
        infos = {
            "time_outs": self.time_out_buf.float(),
            "task_rewards": task_reward,
            "aux_rewards": auxiliary_reward,
            "amp_reward_multiplier": torch.ones_like(rewards),
        }
        return self._obs, self._privileged, rewards, done_buf, infos

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.cat([env.episode_length_buf for env in self.lab_envs])

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        cursor = 0
        for env, size in zip(self.lab_envs, self.pool_sizes, strict=True):
            env.episode_length_buf.copy_(value[cursor : cursor + size])
            cursor += size

    @property
    def commands(self) -> torch.Tensor:
        return torch.cat(
            [env.command_manager.get_command("twist") for env in self.lab_envs]
        )

    @property
    def base_lin_vel(self) -> torch.Tensor:
        return torch.cat(
            [env.scene["robot"].data.root_link_lin_vel_b for env in self.lab_envs]
        )

    @property
    def base_ang_vel(self) -> torch.Tensor:
        return torch.cat(
            [env.scene["robot"].data.root_link_ang_vel_b for env in self.lab_envs]
        )

    @property
    def root_states(self) -> torch.Tensor:
        """Expose Gym-compatible root-state columns for shared diagnostics only."""
        states = []
        for env in self.lab_envs:
            data = env.scene["robot"].data
            position = data.root_link_pos_w - env.scene.env_origins
            states.append(
                torch.cat(
                    (
                        position,
                        data.root_link_quat_w,
                        data.root_link_lin_vel_w,
                        data.root_link_ang_vel_w,
                    ),
                    dim=-1,
                )
            )
        return torch.cat(states)

    @property
    def dof_vel(self) -> torch.Tensor:
        return torch.cat([env.scene["robot"].data.joint_vel for env in self.lab_envs])

    @property
    def base_quat(self) -> torch.Tensor:
        return torch.cat(
            [env.scene["robot"].data.root_link_quat_w for env in self.lab_envs]
        )

    @property
    def rpy(self) -> torch.Tensor:
        from .robot.robot_runtime import _quat_to_rpy_wxyz

        return _quat_to_rpy_wxyz(self.base_quat)

    @property
    def payload_masses(self) -> torch.Tensor:
        return torch.cat([r.payload_mass for r in self.runtimes])

    @property
    def base_com_offsets(self) -> torch.Tensor:
        return torch.cat([r.base_com_offset for r in self.runtimes])

    @property
    def multiplied_link_masses_ratio(self) -> torch.Tensor:
        return torch.cat([r.link_mass_ratio for r in self.runtimes])

    @property
    def motor_strengths(self) -> torch.Tensor:
        return torch.cat([r.motor_strength for r in self.runtimes])

    @property
    def motor_zero_offsets(self) -> torch.Tensor:
        return torch.cat([r.motor_zero_offsets for r in self.runtimes])

    @property
    def friction_coeffs(self) -> torch.Tensor:
        return torch.cat([r.friction for r in self.runtimes])

    @property
    def restitution_coeffs(self) -> torch.Tensor:
        return torch.cat([r.restitution for r in self.runtimes])

    def get_observations(self) -> torch.Tensor:
        return self._obs

    def get_privileged_observations(self) -> torch.Tensor:
        return self._privileged

    def get_amp_observations(self) -> torch.Tensor:
        return torch.cat([runtime.amp_observation() for runtime in self.runtimes])

    def close(self) -> None:
        for env in self.lab_envs:
            env.close()

    @property
    def seesaw_runtime(self):
        return self.runtimes[0]

    def _task_controllers(self):
        return tuple(self.runtimes)

    def _refresh_pool_metadata(self):
        runtime = self.seesaw_runtime
        self.terrain_ids = torch.full(
            (self.num_envs,), 10, dtype=torch.long, device=self.device
        )
        self.terrain_levels = runtime.levels
        self.recovery_reset_type = runtime.recovery_reset_type
        self.disturbance_wrench_timer = runtime.disturbance_wrench_timer
        self.external_wrenches = runtime.external_wrenches

    def set_training_iteration(self, iteration):
        self.seesaw_runtime.update_training_iteration(iteration)
        if not self.freeze_curriculum:
            self.seesaw_runtime.board_curriculum.update()

    def curriculum_state_dict(self):
        if not self.freeze_curriculum:
            self.seesaw_runtime.board_curriculum.update()
        return dict(
            version=1,
            stage=3,
            rank=self.rank,
            pools={},
            mjlab_seesaw=self.seesaw_runtime.board_curriculum.state_dict(),
        )

    def load_curriculum_state_dict(self, state):
        self.seesaw_runtime.board_curriculum.load_state_dict(state["mjlab_seesaw"])

    def _domain_rand_progress(self):
        from .robot.robot_randomization import progress

        return progress(self.cfg.domain_rand, self.seesaw_runtime.training_iteration)

    def _recovery_configured(self):
        return False

    def _recovery_enabled(self):
        return False

    def _walk_recovery_enabled(self):
        return False

    def step(self, actions):
        result = self._step(actions)
        self.time_out_buf |= self.seesaw_runtime.last_success_event
        result[4]["time_outs"] = self.time_out_buf.float()
        return result
