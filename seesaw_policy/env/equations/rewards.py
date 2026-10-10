"""privileged Teacher task rewards ported from the captured Go2 Gym recipes.

This module has no simulator imports.  The Isaac Lab task supplies measured
state tensors and terrain-event terms; this module applies the captured raw
reward equations, per-term weights, control-dt scaling, and task/aux split.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import torch
from .config import ConfigView


@dataclass(frozen=True)
class RewardBatch:
    total: torch.Tensor
    task: torch.Tensor
    auxiliary: torch.Tensor
    terms: dict[str, torch.Tensor]
    raw_terms: dict[str, torch.Tensor]
    invalid_rows: torch.Tensor | None = None


class RobotRewardModel:
    """Evaluate recipe-enabled reward terms from a checked task-state mapping.

    Terrain-specific and Stage3/4 event terms are required as named inputs.
    They are never replaced with zeros when the recipe enables them.
    """

    def __init__(
        self,
        recipe: ConfigView,
        *,
        decomposition_atol: float = 1e-05,
        decomposition_rtol: float = 0.0,
    ):
        self.config = recipe
        cfg = recipe.env["rewards"]
        self.cfg = cfg
        self.decomposition_atol = float(decomposition_atol)
        self.decomposition_rtol = float(decomposition_rtol)
        if self.decomposition_atol < 0.0 or self.decomposition_rtol < 0.0:
            raise ValueError("Reward decomposition tolerances must be nonnegative")
        scales = cfg["scales"]
        weights = cfg.get("weights", {})
        self.scales = {
            name: float(scale) * float(weights.get(name, 1.0))
            for name, scale in scales.items()
            if float(scale) != 0.0
            and float(scale) * float(weights.get(name, 1.0)) != 0.0
        }
        self.recovery_scales = {
            name: float(scale)
            for name, scale in cfg.get("walk_recovery_scales", {}).items()
            if float(scale) != 0.0
        }
        self.events = set(cfg.get("event_reward_names", ()))
        self.task_names = set(cfg.get("task_reward_names", ())) | {
            "recovery_flip",
            "recovery_height",
            "recovery_height_progress",
            "recovery_upright",
            "recovery_tilt_cost",
            "recovery_height_cost",
            "recovery_outcome",
        }
        self.dt = recipe.control_dt

    @staticmethod
    def _need(
        state: Mapping[str, Any], name: str, *, ndim: int | None = None
    ) -> torch.Tensor:
        value = state.get(name)
        if not torch.is_tensor(value):
            raise ValueError(f"reward state is missing tensor {name!r}")
        if ndim is not None and value.ndim != ndim:
            raise ValueError(
                f"reward state {name!r} must have rank {ndim}, got {value.ndim}"
            )
        return value

    def _raw(self, state: Mapping[str, Any], name: str, rows: int) -> torch.Tensor:
        cfg = self.cfg
        z = lambda: torch.zeros(
            rows, device=self._need(state, "actions", ndim=2).device
        )
        scalar = lambda key, default: float(cfg.get(key, default))
        if name == "action_rate":
            return (
                (
                    self._need(state, "last_actions", ndim=2)
                    - self._need(state, "actions", ndim=2)
                )
                .square()
                .sum(1)
            )
        if name == "action_smoothness":
            a = self._need(state, "actions", ndim=2)
            return (
                (
                    a
                    - 2.0 * self._need(state, "last_actions", ndim=2)
                    + self._need(state, "last_last_actions", ndim=2)
                )
                .square()
                .sum(1)
            )
        if name == "ang_vel_xy":
            return self._need(state, "base_ang_vel", ndim=2)[:, :2].square().sum(1)
        if name == "lin_vel_z":
            value = self._need(state, "base_lin_vel", ndim=2)[:, 2]
            clip = cfg.get("lin_vel_z_clip")
            if clip is not None:
                value = value.clamp(-float(clip), float(clip))
            return value.square()
        if name in ("base_acc", "roll_ang_acc"):
            angular_weight = scalar("base_acc_angular_weight", 0.25)
            recovery = state.get("recovery_reward_active")
            if recovery is not None and "recovery_base_acc_angular_weight" in cfg:
                angular_weight = torch.where(
                    recovery.bool(),
                    scalar("recovery_base_acc_angular_weight", 0.25),
                    angular_weight,
                )[:, None]
            linear_acc = (
                self._need(state, "base_lin_vel", ndim=2)
                - self._need(state, "last_base_lin_vel", ndim=2)
            ) / self.dt
            angular_acc = (
                self._need(state, "base_ang_vel", ndim=2)
                - self._need(state, "last_base_ang_vel", ndim=2)
            ) / self.dt
            if bool(cfg.get("world_frame_acceleration", False)):
                linear_acc = (
                    self._need(state, "root_lin_vel_w", ndim=2)
                    - self._need(state, "last_root_lin_vel_w", ndim=2)
                ) / self.dt
                angular_acc = (
                    self._need(state, "root_ang_vel_w", ndim=2)
                    - self._need(state, "last_root_ang_vel_w", ndim=2)
                ) / self.dt
                valid = (self._need(state, "episode_length", ndim=1) > 1)[:, None]
                linear_acc = torch.where(
                    valid, linear_acc, torch.zeros_like(linear_acc)
                )
                angular_acc = torch.where(
                    valid, angular_acc, torch.zeros_like(angular_acc)
                )

            def robust(value: torch.Tensor, delta: float) -> torch.Tensor:
                if not bool(cfg.get("robust_acceleration", False)):
                    return value.square()
                value = torch.nan_to_num(
                    value, nan=10 * delta, posinf=10 * delta, neginf=-10 * delta
                )
                value = value.abs().clamp(max=10 * delta)
                return torch.where(
                    value <= delta, value.square(), 2 * delta * value - delta**2
                )

            if name == "roll_ang_acc":
                return (
                    self._need(state, "base_ang_vel", ndim=2)[:, 0]
                    - self._need(state, "last_base_ang_vel", ndim=2)[:, 0]
                ).square()
            linear = robust(linear_acc, 20.0).sum(1, keepdim=True)
            angular = angular_weight * robust(angular_acc, 50.0)
            self._acceleration_parts = torch.cat((linear, angular), dim=1)
            return self._acceleration_parts.sum(1)
        if name == "roll_orientation":
            return self._need(state, "roll", ndim=1).square()
        if name == "roll_level":
            sigma = max(scalar("roll_orientation_sigma", 0.1), 1e-06)
            return torch.exp(-self._need(state, "roll", ndim=1).square() / sigma)
        if name == "tracking_lin_vel":
            error_sq = self._need(state, "tracking_linear_error_sq", ndim=2)
            sigmas = state.get("tracking_linear_sigmas")
            if sigmas is None:
                sigma = max(scalar("tracking_sigma", 0.1), 1e-06)
                return torch.exp(-error_sq.sum(1) / sigma)
            if sigmas.shape != error_sq.shape:
                raise ValueError(
                    "tracking_linear_sigmas must match tracking_linear_error_sq"
                )
            return torch.exp(-(error_sq / sigmas).sum(1))
        if name == "tracking_ang_vel":
            error_sq = self._need(state, "tracking_angular_error_sq", ndim=1)
            sigma = state.get("tracking_angular_sigma")
            if sigma is None:
                sigma = max(scalar("tracking_sigma", 0.1), 1e-06)
            return torch.exp(-error_sq / sigma)
        if name in ("default_base_height", "correct_base_height"):
            base_height = self._need(state, "base_height", ndim=1)
            target = scalar("base_height_target", 0.35)
            if name == "correct_base_height":
                error_clip = scalar("base_height_error_clip", float("inf"))
                return (base_height - target).clamp(-error_clip, error_clip).square()
            sigma = max(scalar("walk_base_height_sigma", 0.18), 1e-06)
            error = (base_height - target) / sigma
            gaussian = torch.exp(-error.square())
            ramp = (base_height / max(target, 1e-06)).clamp(0.0, 1.0)
            ramp_weight = scalar("turn_over_base_height_ramp_weight", 0.5)
            gaussian_weight = scalar("turn_over_base_height_gaussian_weight", 0.5)
            total_weight = max(ramp_weight + gaussian_weight, 1e-06)
            return (ramp_weight * ramp + gaussian_weight * gaussian) / total_weight
        if name == "dof_pos_limits":
            pos = self._need(state, "joint_pos", ndim=2)
            limits = self._need(state, "joint_pos_limits", ndim=3)
            return (
                -(pos - limits[..., 0]).clamp(max=0.0)
                + (pos - limits[..., 1]).clamp(min=0.0)
            ).sum(1)
        if name == "dof_vel_limits":
            vel = self._need(state, "joint_vel", ndim=2).abs()
            limit = self._need(state, "joint_vel_limits", ndim=2)
            soft = scalar("soft_dof_vel_limit", 0.8)
            return (vel - limit * soft).clamp(min=0.0, max=1.0).sum(1)
        if name == "torque_limits":
            torque = self._need(state, "torques", ndim=2).abs()
            limit = self._need(state, "torque_limits", ndim=2)
            soft = scalar("soft_torque_limit", 0.9)
            return (torque - limit * soft).clamp(min=0.0).sum(1)
        if name == "torques":
            return self._need(state, "torques", ndim=2).square().sum(1)
        if name == "collision":
            force = self._need(state, "penalized_contact_norm", ndim=2)
            threshold = scalar("collision_contact_threshold", 0.1)
            return (force > threshold).float().sum(1)
        if name == "base_contact":
            force = self._need(state, "termination_contact_norm", ndim=2)
            return (force > 1.0).any(1).float()
        if name == "contact_forces":
            force = self._need(state, "contact_force_norm", ndim=2)
            return (force - scalar("max_contact_force", 147.0)).clamp(min=0.0).sum(1)
        if name == "feet_air_time":
            return self._need(state, "feet_air_time_reward", ndim=1)
        if name == "feet_slip":
            contact = self._need(state, "feet_slip_contact", ndim=2).bool()
            velocity = self._need(state, "feet_vel_w", ndim=3)[..., :2]
            return (torch.linalg.vector_norm(velocity, dim=-1) * contact.float()).sum(1)
        if name == "feet_vertical_impact":
            velocity = self._need(state, "feet_vel_w", ndim=3)[..., 2]
            threshold = scalar("feet_vertical_impact_velocity_threshold", 0.15)
            downward = torch.relu(-velocity - threshold).clamp(max=1.5)
            contact = self._need(state, "feet_contact", ndim=2).bool()
            return (downward.square() * contact.float()).sum(1)
        if name == "feet_foothold_edge":
            roughness = self._need(state, "feet_roughness", ndim=2)
            threshold = max(scalar("feet_foothold_roughness_threshold", 0.01), 1e-06)
            score = ((roughness - threshold) / threshold).clamp(0.0, 1.0)
            contact = self._need(state, "feet_slip_contact", ndim=2).bool()
            return (score * contact.float()).sum(1)
        if name == "feet_stair_face":
            score = state.get("feet_stair_face_score")
            if score is None:
                external_terms = state.get("external_terms", {})
                if isinstance(external_terms, Mapping):
                    score = external_terms.get(name)
            if not torch.is_tensor(score) or score.ndim != 1:
                raise ValueError(
                    "reward state must provide a rank-one feet_stair_face score"
                )
            return score
        if name == "legs_distance":
            feet = self._need(state, "feet_pos_base", ndim=3)
            minimum = scalar("min_legs_distance", 0.18)
            front = (minimum - (feet[:, 0, 1] - feet[:, 1, 1])).clamp(min=0.0).square()
            rear = (minimum - (feet[:, 2, 1] - feet[:, 3, 1])).clamp(min=0.0).square()
            return front + rear
        if name == "pose_walk":
            delta = self._need(state, "joint_pos", ndim=2) - self._need(
                state, "default_joint_pos", ndim=2
            )
            command = self._need(state, "commands", ndim=2)
            xy_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
            xy_gate = (
                (scalar("pose_stand_xy_threshold", 0.07) - xy_speed)
                / max(scalar("pose_stand_xy_blend_width", 0.02), 1e-06)
            ).clamp(0.0, 1.0)
            yaw_gate = (
                (scalar("pose_stand_yaw_threshold", 0.1) - command[:, 2].abs())
                / max(scalar("pose_stand_yaw_blend_width", 0.05), 1e-06)
            ).clamp(0.0, 1.0)
            return delta.abs().sum(1) * (1.0 - xy_gate * yaw_gate)
        if name == "pose_stand":
            delta = self._need(state, "joint_pos", ndim=2) - self._need(
                state, "default_joint_pos", ndim=2
            )
            command = self._need(state, "commands", ndim=2)
            xy_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
            xy_gate = (
                (scalar("pose_stand_xy_threshold", 0.07) - xy_speed)
                / max(scalar("pose_stand_xy_blend_width", 0.02), 1e-06)
            ).clamp(0.0, 1.0)
            yaw_gate = (
                (scalar("pose_stand_yaw_threshold", 0.1) - command[:, 2].abs())
                / max(scalar("pose_stand_yaw_blend_width", 0.05), 1e-06)
            ).clamp(0.0, 1.0)
            return delta.abs().sum(1) * xy_gate * yaw_gate
        if name.startswith("walk_hip_"):
            joint_name = {
                "walk_hip_fl": "FL_hip_joint",
                "walk_hip_fr": "FR_hip_joint",
                "walk_hip_rl": "RL_hip_joint",
                "walk_hip_rr": "RR_hip_joint",
            }.get(name)
            if joint_name is None:
                raise ValueError(f"unsupported privileged hip reward {name!r}")
            order = state.get("joint_names")
            if not isinstance(order, (list, tuple)):
                raise ValueError(
                    "reward state must include joint_names in articulation order"
                )
            idx = order.index(joint_name)
            return (
                self._need(state, "joint_pos", ndim=2)[:, idx]
                - self._need(state, "default_joint_pos", ndim=2)[:, idx]
            ).abs()
        if name == "stand_feet_no_contact":
            contact = self._need(state, "feet_contact", ndim=2).bool()
            command = self._need(state, "commands", ndim=2)
            standing = (command[:, :3] == 0.0).all(1)
            return (~contact).sum(1).float() * standing.float()
        if name == "termination":
            failed = self._need(state, "terminated", ndim=1).bool()
            timed_out = self._need(state, "time_out", ndim=1).bool()
            return (failed & ~timed_out).float()
        if name in state.get("external_terms", {}):
            value = state["external_terms"][name]
            if not torch.is_tensor(value) or value.shape != (rows,):
                raise ValueError(f"external reward {name!r} must have shape {(rows,)}")
            return value
        raise NotImplementedError(
            f"recipe {self.config.stage} enables reward {name!r}, but its task event is not wired"
        )

    def _compute_batch(
        self, state: Mapping[str, Any], *, quarantine_nonfinite: bool = False
    ) -> tuple[RewardBatch, torch.Tensor]:
        actions = self._need(state, "actions", ndim=2)
        rows = actions.shape[0]
        raw_terms: dict[str, torch.Tensor] = {}
        weighted_terms: dict[str, torch.Tensor] = {}
        term_multipliers = state.get("term_multipliers", {})
        recovery_mask = state.get("recovery_reward_active")
        if recovery_mask is not None:
            if not torch.is_tensor(recovery_mask) or recovery_mask.shape != (rows,):
                raise ValueError(
                    "recovery_reward_active must be a boolean vector of length rows"
                )
            recovery_mask = recovery_mask.bool()
        task = torch.zeros(rows, dtype=actions.dtype, device=actions.device)
        auxiliary = torch.zeros_like(task)
        invalid_rows = torch.zeros(rows, dtype=torch.bool, device=actions.device)
        termination: torch.Tensor | None = None
        for name in sorted(set(self.scales) | set(self.recovery_scales)):
            raw = self._raw(state, name, rows)
            if raw.shape != (rows,) or raw.device != actions.device:
                raise ValueError(
                    f"reward {name!r} must be a device-local vector of length {rows}"
                )
            raw_terms[name] = raw
            scale = self.scales.get(name, 0.0)
            if name not in self.events:
                scale *= self.dt
            weighted = raw * scale
            if name == "termination":
                termination = weighted
                continue
            if recovery_mask is not None:
                recovery_scale = self.recovery_scales.get(name, 0.0)
                if name not in (
                    "recovery_flip",
                    "recovery_height_progress",
                    "recovery_outcome",
                ):
                    recovery_scale *= self.dt
                weighted = torch.where(recovery_mask, raw * recovery_scale, weighted)
            multiplier = (
                term_multipliers.get(name, 1.0)
                if isinstance(term_multipliers, Mapping)
                else 1.0
            )
            if torch.is_tensor(multiplier):
                if multiplier.shape != (rows,) or multiplier.device != actions.device:
                    raise ValueError(
                        f"term multiplier for {name!r} must be a device-local vector"
                    )
                weighted = weighted * multiplier
            else:
                weighted = weighted * float(multiplier)
            if name == "base_acc":
                factor = (
                    torch.where(
                        raw > 0, weighted / raw.clamp_min(1e-12), torch.zeros_like(raw)
                    )
                    / self.dt
                )
                rates = self._acceleration_parts * factor[:, None]
                for index, axis in enumerate(("linear", "roll", "pitch", "yaw")):
                    raw_terms["_diagnostic_base_acc_" + axis + "_rate"] = rates[
                        :, index
                    ]
                raw_terms["_diagnostic_base_acc_angular_rate"] = rates[:, 1:].sum(1)
                active = (
                    recovery_mask
                    if recovery_mask is not None
                    else torch.zeros_like(raw, dtype=torch.bool)
                )
                raw_terms["_diagnostic_base_acc_recovery_mask"] = active.float()
            weighted_terms[name] = weighted
            if quarantine_nonfinite:
                invalid_rows |= ~torch.isfinite(raw) | ~torch.isfinite(weighted)
            if name in self.task_names:
                task = task + weighted
            else:
                auxiliary = auxiliary + weighted
        total = task + auxiliary
        if "parkour_terms" in state:
            parkour_terms = state["parkour_terms"]
            if not isinstance(parkour_terms, Mapping):
                raise ValueError(
                    "parkour_terms must be a mapping of named reward vectors"
                )
            for name, value in parkour_terms.items():
                if (
                    not torch.is_tensor(value)
                    or value.shape != (rows,)
                    or value.device != actions.device
                ):
                    raise ValueError(
                        f"parkour term {name!r} must be a device-local vector of length {rows}"
                    )
                weighted_terms[name] = value
                if quarantine_nonfinite:
                    invalid_rows |= ~torch.isfinite(value)
                task = task + value
                total = total + value
        if bool(self.cfg.get("only_positive_rewards", False)):
            total = total.clamp(min=0.0)
            auxiliary = total - task
        if termination is not None:
            total = total + termination
            if "termination" in self.task_names:
                task = task + termination
            else:
                auxiliary = auxiliary + termination
            weighted_terms["termination"] = termination
        if quarantine_nonfinite:
            invalid_rows |= ~(
                torch.isfinite(task) & torch.isfinite(auxiliary) & torch.isfinite(total)
            )
            task = task.masked_fill(invalid_rows, 0.0)
            auxiliary = auxiliary.masked_fill(invalid_rows, 0.0)
            total = total.masked_fill(invalid_rows, 0.0)
            raw_terms = {
                name: value.masked_fill(invalid_rows, 0.0)
                for name, value in raw_terms.items()
            }
            weighted_terms = {
                name: value.masked_fill(invalid_rows, 0.0)
                for name, value in weighted_terms.items()
            }
        difference = (task + auxiliary - total).abs()
        tolerance = self.decomposition_atol + self.decomposition_rtol * total.abs()
        bad = ~torch.isfinite(difference) | (difference > tolerance)
        return (
            RewardBatch(
                total,
                task,
                auxiliary,
                weighted_terms,
                raw_terms,
                invalid_rows if quarantine_nonfinite else None,
            ),
            bad,
        )

    def compute(
        self, state: Mapping[str, Any], *, quarantine_nonfinite: bool = False
    ) -> RewardBatch:
        batch, bad = self._compute_batch(
            state, quarantine_nonfinite=quarantine_nonfinite
        )
        total, task, auxiliary = (batch.total, batch.task, batch.auxiliary)
        if bool(bad.any()):
            difference = (task + auxiliary - total).abs()
            row = int(bad.nonzero(as_tuple=False)[0, 0].item())
            raise RuntimeError(
                f"privileged reward task/auxiliary split does not sum to total: row={row}, task={task[row].item()}, auxiliary={auxiliary[row].item()}, total={total[row].item()}, abs_error={difference[row].item()}, finite=({torch.isfinite(task[row]).item()}, {torch.isfinite(auxiliary[row]).item()}, {torch.isfinite(total[row]).item()})"
            )
        return batch
