"""Low-overhead training diagnostics for CTS/MoE experiments.

The simulator and PPO update run on the hot path, so this module deliberately
does not write files from ``env.step``.  ``RolloutMetricsAccumulator`` keeps
small GPU-side moments/counters and ``TrainingMonitor`` serializes one compact
record at the end of an iteration.  The monitor is intentionally independent
of TensorBoard; the runner can publish the same flattened values to both
backends without coupling the training algorithm to a logging format.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch


SCHEMA_VERSION = 2


def _sample_finite(values: torch.Tensor, max_samples: int) -> tuple[torch.Tensor, int, int]:
    """Return a deterministic, evenly-spaced finite sample and source counts."""
    flat = values.detach().float().reshape(-1)
    total_count = int(flat.numel())
    if total_count == 0:
        return flat, 0, 0
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    flat = flat[finite_mask]
    if flat.numel() > max_samples:
        # A prefix over-represents early timesteps and low env ids.  Evenly
        # spaced indices retain coverage without a random-number dependency.
        indices = torch.linspace(
            0,
            flat.numel() - 1,
            steps=max_samples,
            device=flat.device,
        ).long()
        flat = flat.index_select(0, indices)
    return flat, total_count, finite_count


@torch.no_grad()
def summarize_distribution(
    values: Optional[torch.Tensor],
    max_samples: int = 20000,
    histogram_bins: int = 32,
) -> Optional[dict[str, Any]]:
    """Create a compact plot-ready distribution summary.

    The returned payload contains moments, robust quantiles and a normalized
    histogram.  It is small enough for JSONL and avoids storing transitions.
    """
    if values is None:
        return None
    sample, total_count, finite_count = _sample_finite(values, max(int(max_samples), 1))
    if total_count == 0:
        return {
            "count": 0,
            "sampled_count": 0,
            "finite_rate": None,
            "histogram": None,
        }
    if sample.numel() == 0:
        return {
            "count": total_count,
            "sampled_count": 0,
            "finite_rate": 0.0,
            "histogram": None,
        }

    quantile_levels = sample.new_tensor([0.01, 0.10, 0.50, 0.90, 0.99])
    quantiles = torch.quantile(sample, quantile_levels)
    q01, q10, q50, q90, q99 = [float(v) for v in quantiles.cpu().tolist()]
    hist_low = q01
    hist_high = q99
    if not math.isfinite(hist_low) or not math.isfinite(hist_high):
        histogram = None
    else:
        if hist_high - hist_low < 1e-12:
            margin = max(abs(hist_low) * 1e-3, 1e-6)
            hist_low -= margin
            hist_high += margin
        bins = max(int(histogram_bins), 4)
        counts = torch.histc(sample.clamp(hist_low, hist_high), bins=bins, min=hist_low, max=hist_high)
        counts = counts / counts.sum().clamp(min=1.0)
        edges = torch.linspace(hist_low, hist_high, bins + 1, device=sample.device)
        histogram = {
            "edges": [float(v) for v in edges.cpu().tolist()],
            "density": [float(v) for v in counts.cpu().tolist()],
        }

    return {
        "count": total_count,
        "sampled_count": int(sample.numel()),
        "finite_rate": finite_count / float(total_count),
        "mean": float(sample.mean().item()),
        "std": float(sample.std(unbiased=False).item()),
        "abs_mean": float(sample.abs().mean().item()),
        "min": float(sample.min().item()),
        "p01": q01,
        "p10": q10,
        "p50": q50,
        "p90": q90,
        "p99": q99,
        "max": float(sample.max().item()),
        "positive_rate": float((sample > 0).float().mean().item()),
        "zero_rate": float((sample == 0).float().mean().item()),
        "histogram": histogram,
    }


def _safe_explained_variance(target: torch.Tensor, error: torch.Tensor) -> Optional[float]:
    target = target.detach().float().reshape(-1)
    error = error.detach().float().reshape(-1)
    finite = torch.isfinite(target) & torch.isfinite(error)
    if int(finite.sum().item()) < 2:
        return None
    target = target[finite]
    error = error[finite]
    target_var = target.var(unbiased=False)
    if float(target_var.item()) < 1e-12:
        return None
    return float((1.0 - error.var(unbiased=False) / target_var).item())


@torch.no_grad()
def build_rollout_diagnostics(
    storage: Any,
    gamma: float,
    task_weight: float,
    aux_weight: float,
    sample_mask: Optional[torch.Tensor] = None,
    max_samples: int = 20000,
) -> dict[str, Any]:
    """Summarize the reward -> GAE -> critic chain before storage is cleared."""
    required = (
        "task_rewards",
        "aux_rewards",
        "task_values",
        "aux_values",
        "task_returns",
        "aux_returns",
        "task_advantages",
        "aux_advantages",
        "advantages",
        "dones",
    )
    if any(getattr(storage, name, None) is None for name in required):
        return {}

    def masked(value: torch.Tensor) -> torch.Tensor:
        if sample_mask is None:
            return value
        mask = sample_mask
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        return value[mask.expand_as(value)]

    task_rewards = storage.task_rewards.detach().float()
    aux_rewards = storage.aux_rewards.detach().float()
    task_values = storage.task_values.detach().float()
    aux_values = storage.aux_values.detach().float()
    task_returns = storage.task_returns.detach().float()
    aux_returns = storage.aux_returns.detach().float()
    task_adv = storage.task_advantages.detach().float()
    aux_adv = storage.aux_advantages.detach().float()
    final_adv = storage.advantages.detach().float()
    task_adv_raw = task_returns - task_values
    aux_adv_raw = aux_returns - aux_values
    task_weighted = task_adv * float(task_weight)
    aux_weighted = aux_adv * float(aux_weight)
    task_error = task_returns - task_values
    aux_error = aux_returns - aux_values

    task_td = torch.empty_like(task_values)
    aux_td = torch.empty_like(aux_values)
    not_terminal = 1.0 - storage.dones.detach().float()
    task_td[:-1] = (
        task_rewards[:-1]
        + float(gamma) * not_terminal[:-1] * task_values[1:]
        - task_values[:-1]
    )
    aux_td[:-1] = (
        aux_rewards[:-1]
        + float(gamma) * not_terminal[:-1] * aux_values[1:]
        - aux_values[:-1]
    )
    # The rollout bootstrap value is not retained in storage.  Mark the final
    # timestep unavailable rather than silently substituting zero.
    task_td[-1].fill_(float("nan"))
    aux_td[-1].fill_(float("nan"))

    distribution_tensors = {
        "reward_total": task_rewards + aux_rewards,
        "reward_task": task_rewards,
        "reward_aux": aux_rewards,
        "value_task": task_values,
        "value_aux": aux_values,
        "return_task": task_returns,
        "return_aux": aux_returns,
        "td_task": task_td,
        "td_aux": aux_td,
        "gae_task_raw": task_adv_raw,
        "gae_aux_raw": aux_adv_raw,
        "gae_task_normalized": task_adv,
        "gae_aux_normalized": aux_adv,
        "gae_task_weighted": task_weighted,
        "gae_aux_weighted": aux_weighted,
        "advantage_final": final_adv,
        "critic_error_task": task_error,
        "critic_error_aux": aux_error,
    }
    distributions = {
        name: summarize_distribution(masked(value), max_samples=max_samples)
        for name, value in distribution_tensors.items()
    }

    task_weighted_flat = masked(task_weighted).reshape(-1)
    aux_weighted_flat = masked(aux_weighted).reshape(-1)
    final_flat = masked(final_adv).reshape(-1)
    finite = (
        torch.isfinite(task_weighted_flat)
        & torch.isfinite(aux_weighted_flat)
        & torch.isfinite(final_flat)
    )
    task_weighted_flat = task_weighted_flat[finite]
    aux_weighted_flat = aux_weighted_flat[finite]
    final_flat = final_flat[finite]
    if final_flat.numel() > 0:
        task_magnitude = task_weighted_flat.abs().mean()
        aux_magnitude = aux_weighted_flat.abs().mean()
        magnitude_total = (task_magnitude + aux_magnitude).clamp(min=1e-8)
        advantage_energy = final_flat.square().mean().clamp(min=1e-8)
        attribution = {
            "task_weight": float(task_weight),
            "aux_weight": float(aux_weight),
            "task_magnitude_share": float((task_magnitude / magnitude_total).item()),
            "aux_magnitude_share": float((aux_magnitude / magnitude_total).item()),
            "task_alignment": float(((task_weighted_flat * final_flat).mean() / advantage_energy).item()),
            "aux_alignment": float(((aux_weighted_flat * final_flat).mean() / advantage_energy).item()),
            "task_aux_cosine": float(
                (
                    (task_weighted_flat * aux_weighted_flat).sum()
                    / (
                        task_weighted_flat.norm().clamp(min=1e-8)
                        * aux_weighted_flat.norm().clamp(min=1e-8)
                    )
                ).item()
            ),
        }
    else:
        attribution = {
            "task_weight": float(task_weight),
            "aux_weight": float(aux_weight),
            "task_magnitude_share": None,
            "aux_magnitude_share": None,
            "task_alignment": None,
            "aux_alignment": None,
            "task_aux_cosine": None,
        }

    terrain_rows: list[dict[str, Any]] = []
    metadata = getattr(storage, "metadata", None)
    if metadata is not None and metadata.shape[-1] > 1:
        terrain_ids = metadata[..., 1].detach().long()
        terrain_mask = sample_mask if sample_mask is not None else torch.ones_like(terrain_ids, dtype=torch.bool)
        for terrain_id in torch.unique(terrain_ids[terrain_mask]):
            group_mask = terrain_mask & (terrain_ids == terrain_id)
            group_task_error = task_error.squeeze(-1)[group_mask]
            group_aux_error = aux_error.squeeze(-1)[group_mask]
            group_task_return = task_returns.squeeze(-1)[group_mask]
            group_aux_return = aux_returns.squeeze(-1)[group_mask]
            row = {
                "terrain_id": int(terrain_id.item()),
                "sample_count": int(group_mask.sum().item()),
                "task_reward_mean": float(task_rewards.squeeze(-1)[group_mask].mean().item()),
                "aux_reward_mean": float(aux_rewards.squeeze(-1)[group_mask].mean().item()),
                "task_gae_mean": float(task_adv.squeeze(-1)[group_mask].mean().item()),
                "task_gae_std": float(task_adv.squeeze(-1)[group_mask].std(unbiased=False).item()),
                "aux_gae_mean": float(aux_adv.squeeze(-1)[group_mask].mean().item()),
                "aux_gae_std": float(aux_adv.squeeze(-1)[group_mask].std(unbiased=False).item()),
                "advantage_abs_mean": float(final_adv.squeeze(-1)[group_mask].abs().mean().item()),
                "task_critic_rmse": float(group_task_error.square().mean().sqrt().item()),
                "aux_critic_rmse": float(group_aux_error.square().mean().sqrt().item()),
                "task_critic_ev": _safe_explained_variance(group_task_return, group_task_error),
                "aux_critic_ev": _safe_explained_variance(group_aux_return, group_aux_error),
            }
            if metadata.shape[-1] > 0:
                row["terrain_level_mean"] = float(metadata[..., 0][group_mask].float().mean().item())
            if metadata.shape[-1] > 4:
                row["command_level_mean"] = float(metadata[..., 4][group_mask].float().mean().item())
            terrain_rows.append(row)

    return {
        "distributions": distributions,
        "advantage_attribution": attribution,
        "terrain_rows": terrain_rows,
    }


def _json_value(value: Any) -> Any:
    """Convert tensors/numpy/config values to finite JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_value(value.detach().float().cpu().item())
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        value = float(value) if isinstance(value, float) else value
        if isinstance(value, float) and not math.isfinite(value):
            # JSON NaN/Infinity is not portable and hides the first failure.
            return None
        return value
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return str(value)


def _flatten_scalars(value: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten scalar-like values, skipping vectors from the JSON scalar map."""
    result: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten_scalars(item, name))
            continue
        if isinstance(item, torch.Tensor):
            if item.numel() != 1:
                continue
            item = item.detach().float().cpu().item()
        elif isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, bool):
            result[name] = float(item)
        elif isinstance(item, (int, float)) and math.isfinite(float(item)):
            result[name] = float(item)
    return result


class RolloutMetricsAccumulator:
    """GPU-side moments and reset counters for one rollout.

    The accumulator observes the state immediately before each action and the
    terminal flags immediately after ``env.step``.  It performs no host sync;
    all counters are converted to Python values only in ``finalize``.
    """

    _STATE_NAMES = (
        "base_linear_speed",
        "base_angular_speed",
        "base_height",
        "base_tilt",
        "dof_speed",
    )

    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._reset_scalars()

    def _reset_scalars(self) -> None:
        scalar = lambda: torch.zeros((), dtype=torch.float32, device=self.device)
        self.step_count = scalar()
        self.sample_count = scalar()
        self.done_count = scalar()
        self.timeout_count = scalar()
        self.fall_count = scalar()
        self.base_contact_count = scalar()
        self.orientation_guard_count = scalar()
        self.velocity_guard_count = scalar()
        self.velocity_nonfinite_count = scalar()
        self.velocity_linear_guard_count = scalar()
        self.velocity_angular_guard_count = scalar()
        self.recovery_timeout_count = scalar()
        self.finite_obs_count = scalar()
        self.finite_action_count = scalar()
        self.finite_reward_count = scalar()
        self.finite_state_count = scalar()
        self.state_sample_count = scalar()
        self.state_sum = torch.zeros(5, dtype=torch.float32, device=self.device)
        self.state_sq_sum = torch.zeros_like(self.state_sum)
        self.state_max = torch.zeros_like(self.state_sum)
        self.action_abs_sum = scalar()
        self.action_abs_sq_sum = scalar()
        self.action_abs_max = scalar()
        self.reward_sum = scalar()
        self.reward_sq_sum = scalar()
        self.task_reward_sum = scalar()
        self.aux_reward_sum = scalar()

    def reset(self) -> None:
        self._reset_scalars()

    @torch.no_grad()
    def observe_before_step(self, obs: torch.Tensor, actions: torch.Tensor, env: Any) -> None:
        """Accumulate pre-action state/action moments without host transfers."""
        self.step_count += 1.0
        self.sample_count += float(self.num_envs)
        obs_finite = torch.isfinite(obs).all(dim=-1)
        action_finite = torch.isfinite(actions).all(dim=-1)
        self.finite_obs_count += obs_finite.float().sum()
        self.finite_action_count += action_finite.float().sum()

        base_lin = getattr(env, "base_lin_vel", None)
        base_ang = getattr(env, "base_ang_vel", None)
        root_states = getattr(env, "root_states", None)
        rpy = getattr(env, "rpy", None)
        dof_vel = getattr(env, "dof_vel", None)
        if base_lin is None or base_ang is None or root_states is None:
            return
        self.state_sample_count += float(self.num_envs)
        base_lin_speed = torch.linalg.vector_norm(base_lin[..., :3], dim=-1)
        base_ang_speed = torch.linalg.vector_norm(base_ang[..., :3], dim=-1)
        base_height = root_states[..., 2]
        if rpy is None:
            base_tilt = torch.zeros_like(base_height)
        else:
            base_tilt = torch.linalg.vector_norm(rpy[..., :2], dim=-1)
        if dof_vel is None:
            dof_speed = torch.zeros_like(base_height)
        else:
            dof_speed = torch.linalg.vector_norm(dof_vel, dim=-1)
        values = torch.stack(
            (base_lin_speed, base_ang_speed, base_height, base_tilt, dof_speed),
            dim=1,
        ).float()
        finite = torch.isfinite(values).all(dim=1)
        self.finite_state_count += finite.float().sum()
        safe_values = torch.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        self.state_sum += safe_values.sum(dim=0)
        self.state_sq_sum += safe_values.square().sum(dim=0)
        self.state_max = torch.maximum(self.state_max, safe_values.abs().amax(dim=0))

        action_abs = torch.linalg.vector_norm(actions.float(), dim=-1)
        action_abs = torch.nan_to_num(action_abs, nan=0.0, posinf=1e6, neginf=0.0)
        self.action_abs_sum += action_abs.sum()
        self.action_abs_sq_sum += action_abs.square().sum()
        self.action_abs_max = torch.maximum(self.action_abs_max, action_abs.max())


    @torch.no_grad()
    def observe_after_step(
        self,
        rewards: torch.Tensor,
        task_rewards: Optional[torch.Tensor],
        aux_rewards: Optional[torch.Tensor],
        dones: torch.Tensor,
        env: Any,
    ) -> None:
        """Accumulate reward/termination diagnostics after ``env.step``."""
        rewards = rewards.float().reshape(-1)
        self.finite_reward_count += torch.isfinite(rewards).float().sum()
        safe_rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1e6, neginf=-1e6)
        self.reward_sum += safe_rewards.sum()
        self.reward_sq_sum += safe_rewards.square().sum()
        if task_rewards is not None:
            self.task_reward_sum += torch.nan_to_num(
                task_rewards.float().reshape(-1), nan=0.0, posinf=1e6, neginf=-1e6
            ).sum()
        if aux_rewards is not None:
            self.aux_reward_sum += torch.nan_to_num(
                aux_rewards.float().reshape(-1), nan=0.0, posinf=1e6, neginf=-1e6
            ).sum()

        done_mask = dones.bool().reshape(-1)
        self.done_count += done_mask.float().sum()
        self.timeout_count += self._event_count(env, "last_timeout_buf", "time_out_buf")
        self.fall_count += self._event_count(env, "last_fall_buf", "episode_had_fall")
        self.base_contact_count += self._event_count(
            env, "last_base_contact_fall_buf", None
        )
        self.orientation_guard_count += self._event_count(
            env, "last_orientation_guard_reset_buf", "orientation_guard_reset_buf"
        )
        self.velocity_guard_count += self._event_count(
            env, "last_velocity_guard_reset_buf", "velocity_guard_reset_buf"
        )
        self.velocity_nonfinite_count += self._event_count(
            env, "last_velocity_nonfinite_reset_buf", "velocity_nonfinite_reset_buf"
        )
        self.velocity_linear_guard_count += self._event_count(
            env, "last_velocity_lin_guard_reset_buf", "velocity_lin_guard_reset_buf"
        )
        self.velocity_angular_guard_count += self._event_count(
            env, "last_velocity_ang_guard_reset_buf", "velocity_ang_guard_reset_buf"
        )
        self.recovery_timeout_count += self._event_count(
            env, "last_recovery_timeout_buf", "recovery_timeout_buf"
        )


    @staticmethod
    def _event_count(env: Any, preferred: str, fallback: Optional[str]) -> torch.Tensor:
        value = getattr(env, preferred, None)
        if value is None and fallback is not None:
            value = getattr(env, fallback, None)
        if value is None:
            return torch.zeros(
                (),
                dtype=torch.float32,
                device=getattr(env, "device", "cpu"),
            )
        return value.detach().float().reshape(-1).sum()

    def finalize(self) -> dict[str, Any]:
        """Return host scalars for JSON/TensorBoard publication."""
        count = self.sample_count.clamp(min=1.0)
        state_mean = self.state_sum / count
        state_var = (self.state_sq_sum / count - state_mean.square()).clamp(min=0.0)
        action_mean = self.action_abs_sum / count
        action_var = (self.action_abs_sq_sum / count - action_mean.square()).clamp(min=0.0)
        reward_mean = self.reward_sum / count
        reward_var = (self.reward_sq_sum / count - reward_mean.square()).clamp(min=0.0)
        denom = count

        def values(tensor: torch.Tensor) -> list[float]:
            return [float(x) for x in tensor.detach().cpu().tolist()]

        result: dict[str, Any] = {
            "rollout_steps": float(self.step_count.item()),
            "rollout_samples": float(self.sample_count.item()),
            "done_count": float(self.done_count.item()),
            "done_rate": float((self.done_count / denom).item()),
            "timeout_rate": float((self.timeout_count / denom).item()),
            "fall_rate": float((self.fall_count / denom).item()),
            "base_contact_rate": float((self.base_contact_count / denom).item()),
            "orientation_guard_rate": float((self.orientation_guard_count / denom).item()),
            "velocity_guard_rate": float((self.velocity_guard_count / denom).item()),
            "velocity_nonfinite_rate": float((self.velocity_nonfinite_count / denom).item()),
            "velocity_linear_guard_rate": float((self.velocity_linear_guard_count / denom).item()),
            "velocity_angular_guard_rate": float((self.velocity_angular_guard_count / denom).item()),
            "recovery_timeout_rate": float((self.recovery_timeout_count / denom).item()),
            "obs_finite_rate": float((self.finite_obs_count / denom).item()),
            "action_finite_rate": float((self.finite_action_count / denom).item()),
            "reward_finite_rate": float((self.finite_reward_count / denom).item()),
            "state_finite_rate": (
                float((self.finite_state_count / self.state_sample_count).item())
                if float(self.state_sample_count.item()) > 0.0
                else None
            ),
            "reward_mean": float(reward_mean.item()),
            "reward_std": float(torch.sqrt(reward_var).item()),
            "task_reward_mean": float((self.task_reward_sum / denom).item()),
            "aux_reward_mean": float((self.aux_reward_sum / denom).item()),
            "action_abs_mean": float(action_mean.item()),
            "action_abs_std": float(torch.sqrt(action_var).item()),
            "action_abs_max": float(self.action_abs_max.item()),
        }
        for index, name in enumerate(self._STATE_NAMES):
            result[f"state_{name}_mean"] = float(state_mean[index].item())
            result[f"state_{name}_std"] = float(torch.sqrt(state_var[index]).item())
            result[f"state_{name}_max"] = float(self.state_max[index].item())
        return result


class TrainingMonitor:
    """Append-only JSONL monitor with lightweight anomaly detection."""

    def __init__(
        self,
        log_dir: str | Path,
        train_cfg: Optional[Mapping[str, Any]] = None,
        env_cfg: Optional[Any] = None,
        snapshot_on_alert: bool = True,
        snapshot_max_envs: int = 8,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.alerts_path = self.log_dir / "alerts.jsonl"
        self._metrics_file = self.metrics_path.open("a", encoding="utf-8", buffering=1)
        self._alerts_file = self.alerts_path.open("a", encoding="utf-8", buffering=1)
        self.snapshot_on_alert = bool(snapshot_on_alert)
        self.snapshot_max_envs = max(int(snapshot_max_envs), 1)
        self._last_alert_iteration: dict[str, int] = {}
        self._metric_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=50))
        self._write_manifest(train_cfg, env_cfg)

    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            commit = result.stdout.strip()
            return commit or None
        except Exception:
            return None

    @staticmethod
    def _git_text(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            value = result.stdout.strip()
            return value or None
        except Exception:
            return None

    def _write_manifest(self, train_cfg: Optional[Mapping[str, Any]], env_cfg: Any) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git_commit": self._git_commit(),
            "git_branch": self._git_text("branch", "--show-current"),
            "git_dirty": self._git_text("status", "--porcelain") is not None,
            "train_config": _json_value(train_cfg or {}),
            "env_config": _json_value(env_cfg),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload["config_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        manifest_path = self.log_dir / "run_manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )

    def _alerts(self, metrics: Mapping[str, float], iteration: int) -> list[dict[str, Any]]:
        def metric(name: str, default: Optional[float] = None) -> Optional[float]:
            value = metrics.get(name, default)
            if value is None:
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else float("nan")

        alerts: list[dict[str, Any]] = []

        def add(
            code: str,
            severity: str,
            name: str,
            value: float,
            threshold: float,
            message: str,
        ) -> None:
            alerts.append({
                "code": code,
                "severity": severity,
                "metric": name,
                "value": None if not math.isfinite(value) else value,
                "threshold": threshold,
                "message": message,
                "iteration": int(iteration),
            })

        for name in ("obs_finite_rate", "action_finite_rate", "reward_finite_rate", "state_finite_rate"):
            value = metric(f"rollout/{name}", 1.0)
            if value is not None and (not math.isfinite(value) or value < 1.0):
                add("nonfinite_rollout", "critical", name, value, 1.0, "non-finite rollout data")

        checks = (
            ("rollout/velocity_nonfinite_rate", ">", 0.0, "critical", "velocity non-finite reset"),
            ("ppo/kl", ">", 0.08, "critical", "policy KL is too large"),
            ("ppo/clip_fraction", ">", 0.50, "critical", "most PPO samples are clipped"),
            ("rollout/fall_rate", ">", 0.90, "warning", "rollout fall rate is high"),
            ("critic/task/explained_variance", "<", 0.0, "warning", "task critic explained variance is non-positive"),
            ("critic/aux/explained_variance", "<", 0.0, "warning", "aux critic explained variance is non-positive"),
            ("moe/dead_experts", ">", 0.5, "warning", "many MoE experts are unused"),
            ("data/terrain_count_min_fraction", "<=", 0.0, "warning", "one or more terrains have no samples"),
        )
        for name, operator, threshold, severity, message in checks:
            value = metric(name)
            if value is None:
                continue
            if operator == ">":
                triggered = value > threshold
            elif operator == "<":
                triggered = value < threshold
            else:
                triggered = value <= threshold
            if triggered:
                add(name.replace("/", "_"), severity, name, value, threshold, message)

        # Robust, scale-independent trend checks.  They only start after a
        # 20-iteration baseline and therefore do not flag normal warmup.
        trend_specs = {
            "critic/task/explained_variance": ("lower", 6.0, "task critic EV dropped from its recent baseline"),
            "critic/aux/explained_variance": ("lower", 6.0, "aux critic EV dropped from its recent baseline"),
            "ppo/kl": ("higher", 8.0, "PPO KL spiked relative to its recent baseline"),
            "gae/final_std": ("higher", 8.0, "final advantage scale spiked"),
            "episode/student_length": ("lower", 8.0, "student episode length dropped sharply"),
        }
        for name, (direction, z_threshold, message) in trend_specs.items():
            value = metric(name)
            history = self._metric_history[name]
            if value is not None and math.isfinite(value) and len(history) >= 20:
                history_tensor = torch.tensor(list(history), dtype=torch.float32)
                center = float(history_tensor.median().item())
                mad = float((history_tensor - center).abs().median().item())
                robust_scale = max(1.4826 * mad, abs(center) * 1e-3, 1e-8)
                robust_z = (value - center) / robust_scale
                triggered = robust_z < -z_threshold if direction == "lower" else robust_z > z_threshold
                if triggered:
                    add(
                        f"trend_{name.replace('/', '_')}",
                        "warning",
                        name,
                        value,
                        center,
                        message,
                    )
            if value is not None and math.isfinite(value):
                history.append(value)
        return alerts

    @staticmethod
    def _nested_value(mapping: Mapping[str, Any], path: str) -> tuple[bool, Any]:
        value: Any = mapping
        for part in path.split("/"):
            if not isinstance(value, Mapping) or part not in value:
                return False, None
            value = value[part]
        return True, value

    def _snapshot_env_ids(self, snapshot: Mapping[str, Any]) -> Optional[torch.Tensor]:
        """Prefer terminal/non-finite envs over an arbitrary tensor prefix."""
        candidate_masks = []
        for name in (
            "reset_buf",
            "last_fall_buf",
            "last_base_contact_fall_buf",
            "last_orientation_guard_reset_buf",
            "last_velocity_guard_reset_buf",
            "last_velocity_nonfinite_reset_buf",
            "last_recovery_timeout_buf",
        ):
            value = snapshot.get(name)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                candidate_masks.append(value.detach().bool().reshape(value.shape[0], -1).any(dim=1))
        for name in ("obs", "privileged_obs", "actions", "rewards"):
            value = snapshot.get(name)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                finite = torch.isfinite(value.detach()).reshape(value.shape[0], -1).all(dim=1)
                candidate_masks.append(~finite)
        if not candidate_masks:
            return None
        size = candidate_masks[0].numel()
        combined = candidate_masks[0].new_zeros(size)
        for candidate in candidate_masks:
            if candidate.numel() == size:
                combined |= candidate
        ids = combined.nonzero(as_tuple=False).reshape(-1)
        if ids.numel() == 0:
            ids = torch.arange(size, device=combined.device)
        return ids[: self.snapshot_max_envs]

    def _compact_snapshot(
        self,
        snapshot: Any,
        env_ids: Optional[torch.Tensor] = None,
    ) -> Any:
        """Copy only selected environment tensors to CPU."""
        if isinstance(snapshot, torch.Tensor):
            value = snapshot.detach()
            if value.ndim > 0:
                if (
                    env_ids is not None
                    and env_ids.numel() > 0
                    and value.shape[0] > int(env_ids.max().item())
                ):
                    value = value.index_select(0, env_ids.to(value.device))
                elif value.shape[0] > self.snapshot_max_envs:
                    value = value[: self.snapshot_max_envs]
            return value.cpu()
        if isinstance(snapshot, Mapping):
            return {str(k): self._compact_snapshot(v, env_ids) for k, v in snapshot.items()}
        if isinstance(snapshot, (tuple, list)):
            return [self._compact_snapshot(v, env_ids) for v in snapshot]
        return snapshot

    def record(
        self,
        iteration: int,
        metrics: Mapping[str, Any],
        snapshot: Optional[Mapping[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Write one iteration record and return the generated alerts."""
        # Convert tensors once.  Flattening the original mapping and then
        # serializing it would force a second GPU->CPU synchronization for
        # every scalar algorithm diagnostic.
        json_metrics = _json_value(metrics)
        scalar_metrics = _flatten_scalars(json_metrics)
        alerts = self._alerts(scalar_metrics, int(iteration))
        # ``_json_value`` maps non-finite tensors to ``None`` so the record is
        # still valid JSON.  Turn that sentinel into an explicit update alert
        # for the core optimization signals instead of silently dropping it
        # from the scalar map.
        for path in (
            "ppo/kl",
            "ppo/ratio_mean",
            "critic/task/loss",
            "critic/aux/loss",
            "distillation/latent_mse",
            "moe/gate_entropy_normalized",
        ):
            present, value = self._nested_value(json_metrics, path)
            if present and value is None:
                alerts.append({
                    "code": "nonfinite_update_metric",
                    "severity": "critical",
                    "metric": path,
                    "value": None,
                    "threshold": 1.0,
                    "message": "non-finite optimization metric",
                    "iteration": int(iteration),
                })
        record = {
            "time": time.time(),
            "iteration": int(iteration),
            "metrics": json_metrics,
            "scalars": scalar_metrics,
            "alerts": alerts,
        }
        self._metrics_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        for alert in alerts:
            code = str(alert["code"])
            last = self._last_alert_iteration.get(code, -10**9)
            if int(iteration) - last < 5:
                continue
            self._last_alert_iteration[code] = int(iteration)
            self._alerts_file.write(json.dumps(alert, ensure_ascii=False, allow_nan=False) + "\n")
        if alerts and snapshot is not None and self.snapshot_on_alert:
            path = self.log_dir / f"debug_iter_{int(iteration):07d}.pt"
            if not path.exists():
                env_ids = self._snapshot_env_ids(snapshot)
                torch.save(
                    {
                        "iteration": int(iteration),
                        "alerts": alerts,
                        "env_ids": env_ids.detach().cpu() if env_ids is not None else None,
                        "snapshot": self._compact_snapshot(snapshot, env_ids),
                    },
                    path,
                )
        return alerts

    def close(self) -> None:
        for file_obj in (self._metrics_file, self._alerts_file):
            try:
                file_obj.flush()
                file_obj.close()
            except Exception:
                pass
