"""Configuration view used by the shared seesaw physics and reward adapters."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfigView:
    """The captured environment and RSL-RL configuration for one stage."""

    stage: str
    source: str
    sha256: str
    manifest: dict[str, Any]

    @property
    def env(self) -> dict[str, Any]:
        return self.manifest["env_config"]

    @property
    def train(self) -> dict[str, Any]:
        return self.manifest["train_config"]

    @property
    def physics_dt(self) -> float:
        return float(self.env["sim"]["dt"])

    @property
    def decimation(self) -> int:
        return int(self.env["control"]["decimation"])

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation

    def dimensions(self) -> dict[str, int]:
        env = self.env["env"]
        return {
            "num_envs": int(env["num_envs"]),
            "num_obs": int(env["num_observations"]),
            "num_privileged_obs": int(env["num_privileged_obs"]),
            "num_actions": int(env["num_actions"]),
        }
