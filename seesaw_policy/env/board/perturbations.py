"""Small, frequent physical pulses for mounted seesaw recovery."""

import math
import torch


def perturbation_config(start_iteration):
    return dict(
        enabled=True,
        start_iteration=int(start_iteration),
        ramp_iterations=1000,
        episode_fraction=0.7,
        mount_grace_s=1.0,
        interval_s=[1.0, 2.0],
        duration_s=[0.1, 0.2],
        board_torque_start_nm=[0.1, 0.3],
        board_torque_end_nm=[0.3, 1.0],
        robot_force_start_n=[1.0, 2.0],
        robot_force_end_n=[2.0, 4.0],
    )


def progress(config, iteration):
    return min(
        1.0,
        max(0.0, (iteration - config["start_iteration"]) / config["ramp_iterations"]),
    )


class SeesawPerturbations:
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        n, device = runtime.num_envs, runtime.device
        self.enabled = torch.zeros(n, dtype=torch.bool, device=device)
        self.wait = torch.zeros(n, 2, device=device)
        self.remaining = torch.zeros_like(self.wait)
        self.duration = torch.ones_like(self.wait)
        self.amplitude = torch.zeros_like(self.wait)
        self.direction = torch.zeros(n, 2, device=device)
        self.board_torque = torch.zeros(n, device=device)
        self.robot_force = torch.zeros(n, 3, device=device)
        self.events = torch.zeros(2, dtype=torch.long, device=device)

    def uniform(self, bounds, shape):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
            shape, device=self.runtime.device, generator=self.runtime._rng
        )

    def reset(self, ids):
        r = self.runtime
        self.enabled[ids] = (
            self.uniform([0.0, 1.0], (len(ids),)) < self.config["episode_fraction"]
        )
        self.wait[ids] = self.uniform(self.config["interval_s"], (len(ids), 2))
        self.remaining[ids] = 0.0
        self.board_torque[ids] = 0.0
        self.robot_force[ids] = 0.0
        # Reset can select a different hinge; clear every old board force in
        # these worlds, without touching robot DOFs or other task worlds.
        r.env.sim.data.qfrc_applied[
            ids[:, None], r.board_randomization.dof_ids[None, :]
        ] = 0.0
        self.write_robot(ids)

    def write_robot(self, ids):
        r = self.runtime
        force = self.robot_force[ids, None, :]
        torque = torch.zeros_like(force)
        r.entity.write_external_wrench_to_sim(
            force, torque, env_ids=ids, body_ids=[r.base_body_id]
        )
        r.external_wrenches[ids, :3] = force[:, 0]
        r.external_wrenches[ids, 3:] = 0.0

    def step(self, dt):
        r, c = self.runtime, self.config
        ids = (
            r.active_mask.nonzero(as_tuple=False).flatten()
            if r.active_mask is not None
            else torch.arange(r.num_envs, device=r.device)
        )
        b = r.boarding
        eligible = (
            self.enabled[ids]
            & b.mounted[ids]
            & b.supported[ids]
            & ~b.failure[ids]
            & (b.mounted_time[ids] >= c["mount_grace_s"])
            & (r.recovery_reset_type[ids] == 0)
        )
        self.remaining[ids] = (self.remaining[ids] - dt).clamp_min(0.0)
        self.remaining[ids[~eligible]] = 0.0
        self.wait[ids] -= eligible[:, None] * dt
        p = progress(c, r.training_iteration)
        for channel, prefix in enumerate(("board_torque", "robot_force")):
            unit = "nm" if channel == 0 else "n"
            low, high = [
                a + p * (z - a)
                for a, z in zip(c[f"{prefix}_start_{unit}"], c[f"{prefix}_end_{unit}"])
            ]
            due = ids[
                eligible
                & (self.wait[ids, channel] <= 0.0)
                & (self.remaining[ids, channel] <= 0.0)
            ]
            self.duration[due, channel] = self.uniform(c["duration_s"], (len(due),))
            self.remaining[due, channel] = self.duration[due, channel]
            self.wait[due, channel] = self.uniform(c["interval_s"], (len(due),))
            self.amplitude[due, channel] = self.uniform([low, high], (len(due),))
            if channel == 0:
                self.amplitude[due, channel] *= torch.where(
                    self.uniform([0.0, 1.0], (len(due),)) < 0.5, -1.0, 1.0
                )
            else:
                angle = self.uniform([-math.pi, math.pi], (len(due),))
                self.direction[due] = torch.stack((angle.cos(), angle.sin()), -1)
            self.events[channel] += len(due)
        # Half-sine pulses start and end at zero, avoiding force discontinuities.
        phase = 1.0 - self.remaining[ids] / self.duration[ids]
        envelope = torch.sin(math.pi * phase).clamp_min(0.0) * (
            self.remaining[ids] > 0.0
        )
        values = self.amplitude[ids] * envelope
        self.board_torque[ids] = values[:, 0]
        self.robot_force[ids, :2] = values[:, 1, None] * self.direction[ids]
        self.robot_force[ids, 2] = 0.0
        r.env.sim.data.qfrc_applied[
            ids, r.board_randomization.dof_ids[r.levels[ids]]
        ] = self.board_torque[ids]
        self.write_robot(ids)

    def metrics(self, ids):
        return dict(
            perturbation_progress=progress(
                self.config, self.runtime.training_iteration
            ),
            perturbation_episode_fraction=float(self.enabled[ids].float().mean()),
            board_torque_abs_mean=float(self.board_torque[ids].abs().mean()),
            robot_push_force_mean=float(self.robot_force[ids].norm(dim=-1).mean()),
            board_torque_events=int(self.events[0]),
            robot_push_events=int(self.events[1]),
        )
