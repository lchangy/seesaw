"""Per-reset board parameters and matching native collision bounds/inertia."""

import mujoco
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from .board_profiles import effective_board_config


def box_inertia(mass, width, length=None):
    length = torch.full_like(width, 3.6) if length is None else length
    return (
        mass[..., None]
        / 12
        * torch.stack(
            (
                width.square() + 0.04**2,
                length.square() + 0.04**2,
                width.square() + length.square(),
            ),
            dim=-1,
        )
    )


def touchdown_angle(height, half_length, half_thickness=0.02):
    radius = (half_length.square() + half_thickness**2).sqrt()
    return torch.asin(height / radius) - torch.atan2(
        torch.full_like(half_length, half_thickness), half_length
    )


def sample_board_parameters(config, count, device, generator, iteration=0):
    effective = effective_board_config(config, iteration)
    ramp = config.get("seesaw3_5")
    base_mask = (
        torch.rand(count, device=device, generator=generator) < ramp["base_fraction"]
        if ramp
        else None
    )
    result = {}
    for name, key in (
        ("mass", "mass_range_kg"),
        ("damping", "damping_range"),
        ("friction", "friction_range"),
        ("width", "width_range_m"),
    ):
        low, high = effective[key]
        unit = torch.rand(count, device=device, generator=generator)
        value = low + (high - low) * unit
        if ramp:
            base_low, base_high = ramp["base_ranges"][key]
            value = torch.where(
                base_mask, base_low + (base_high - base_low) * unit, value
            )
        result[name] = value
    if "length_range_m" in config:
        low, high = effective["length_range_m"]
        result["length"] = low + (high - low) * torch.rand(
            count, device=device, generator=generator
        )
    return result


class BoardDomainRandomization:
    def __init__(self, runtime):
        self.runtime = runtime
        model = runtime.env.sim.mj_model
        self.board_geoms = torch.tensor(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, f"seesaw_l{i}/board_box"
                )
                for i in range(10)
            ],
            device=runtime.device,
        )
        self.support_geoms = torch.tensor(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, f"seesaw_l{i}/support_box"
                )
                for i in range(10)
            ],
            device=runtime.device,
        )
        entities = [runtime.env.scene[f"seesaw_l{i}"] for i in range(10)]
        self.body_ids = torch.tensor(
            [e.indexing.body_ids[e.body_names.index("board")] for e in entities],
            device=runtime.device,
        )
        self.dof_ids = torch.tensor(
            [e.indexing.joint_v_adr[0] for e in entities], device=runtime.device
        )
        self.joint_ids = torch.tensor(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"seesaw_l{i}/hinge"
                )
                for i in range(10)
            ],
            device=runtime.device,
        )

    def apply(self, ids):
        r = self.runtime
        config = r.config.get("domain_rand", {})
        if not config.get("enabled", False):
            return
        values = sample_board_parameters(
            config, len(ids), r.device, r._rng, r.training_iteration
        )
        inspection = getattr(r, "play_board_geometry", None)
        if inspection is not None:
            values["length"] = torch.full_like(values["width"], inspection["length"])
            values["width"] = torch.full_like(values["width"], inspection["width"])
        r.board_half_width[ids] = values["width"] / 2
        length = values.get("length", torch.full_like(values["width"], 3.6))
        r.board_half_length[ids] = length / 2
        r.board_mass[ids] = values["mass"]
        r.board_damping[ids] = values["damping"]
        r.board_friction[ids] = values["friction"]
        model = r.env.sim.model
        levels = r.levels[ids]
        geoms, bodies, dofs = (
            self.board_geoms[levels],
            self.body_ids[levels],
            self.dof_ids[levels],
        )
        model.body_mass[ids, bodies] = values["mass"]
        model.body_inertia[ids, bodies] = box_inertia(
            values["mass"], values["width"], length
        )
        model.dof_damping[ids, dofs] = values["damping"]
        model.geom_friction[ids, geoms, 0] = values["friction"]
        model.geom_size[ids, geoms, 1] = values["width"] / 2
        model.geom_size[ids, geoms, 0] = length / 2
        if "length" in values:
            from seesaw_policy.env.boarding.parameters import ANGLES_DEG, lookup

            angle = (
                (
                    lookup(ANGLES_DEG, levels)
                    if inspection is None
                    else torch.full_like(length, inspection["angle_deg"])
                )
                * torch.pi
                / 180
            )
            height = length / 2 * angle.sin() + 0.02 * angle.cos()
            r.board_height[ids] = height
            r.board_max_angle[ids] = touchdown_angle(height, length / 2)
            model.jnt_range[ids, self.joint_ids[levels]] = torch.stack(
                (-r.board_max_angle[ids], r.board_max_angle[ids]), -1
            )
            model.body_pos[ids, bodies, 2] = height
            model.geom_size[ids, self.support_geoms[levels], 2] = height / 2
            model.geom_pos[ids, self.support_geoms[levels], 2] = height / 2
        model.geom_size[ids, self.support_geoms[levels], 1] = values["width"] * 0.3
        # MuJoCo Warp broadphase and raycasting must see the resized box too.
        for selected in (geoms, self.support_geoms[levels]):
            half = model.geom_size[ids, selected]
            model.geom_rbound[ids, selected] = half.norm(dim=-1)
            model.geom_aabb[ids, selected, 1] = half


@requires_model_fields(
    "body_mass",
    "body_inertia",
    "body_pos",
    "geom_pos",
    "jnt_range",
    "dof_damping",
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "geom_friction",
    "geom_solref",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    recompute=RecomputeLevel.set_const,
)
def seesaw_reset_event(env, env_ids):
    runtime = getattr(env, "task_runtime", None)
    if runtime is not None:
        runtime.reset(env_ids)
