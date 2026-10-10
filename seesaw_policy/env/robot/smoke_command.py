"""Small flat-smoke velocity command; independent of mjlab's task registry.

The full staged command curriculum is not implemented here. Keeping this
local avoids importing mjlab's bundled RSL-RL runner into the project fork.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


class UniformVelocityCommand(CommandTerm):
    def __init__(self, cfg: "UniformVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        scratch = torch.empty(len(env_ids), device=self.device)
        for axis, bounds in enumerate(
            (
                self.cfg.ranges.lin_vel_x,
                self.cfg.ranges.lin_vel_y,
                self.cfg.ranges.ang_vel_z,
            )
        ):
            self.vel_command_b[env_ids, axis] = scratch.uniform_(*bounds)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
    entity_name: str

    @dataclass
    class Ranges:
        lin_vel_x: tuple[float, float]
        lin_vel_y: tuple[float, float]
        ang_vel_z: tuple[float, float]

    ranges: Ranges

    def build(self, env) -> UniformVelocityCommand:
        return UniformVelocityCommand(self, env)
