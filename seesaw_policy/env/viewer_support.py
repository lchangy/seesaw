from __future__ import annotations
import torch


class _ViewerEnv:
    """Expose one MJLab task pool to the viewer without changing privileged stepping."""

    def __init__(self, task_env):
        self.task_env = task_env
        self.display_pool_index = 0
        self.display_offset = 0
        self.unwrapped = task_env.lab_envs[0]
        self.cfg = self.unwrapped.cfg
        self.device = task_env.device
        self.num_envs = task_env.num_envs

    def get_observations(self) -> torch.Tensor:
        return self.task_env.get_observations()

    def step(self, actions: torch.Tensor):
        result = self.task_env.step(actions)
        return result

    def reset(self):
        result = self.task_env.reset()
        return result

    def close(self) -> None:
        self.task_env.close()
