"""Frozen configuration for the standalone seesaw scene only."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskConfig:
    stage: int
    source_sha256: str
    recipe_sha256: str
    env_config: dict[str, Any]
    train_config: dict[str, Any]
    source: str

    @property
    def num_envs(self) -> int:
        return int(self.env_config["env"]["num_envs"])

    @property
    def num_steps_per_env(self) -> int:
        return int(self.train_config["runner"]["num_steps_per_env"])

    @property
    def control_dt(self) -> float:
        return float(self.env_config["sim"]["dt"]) * int(
            self.env_config["control"]["decimation"]
        )


def load_task_config(*, physics_dt=None):
    path = Path(__file__).resolve().parents[3] / "seesaw_policy/task.json"
    recipe = TaskConfig(**json.loads(path.read_text()))
    if physics_dt is not None and physics_dt != recipe.env_config["sim"]["dt"]:
        raise ValueError("Use the seesaw task physics timestep")
    return recipe
