"""Go2 mjlab entity backed by Unitree's official unitree_mujoco Go2 MJCF.

This is a physics asset adapter, not a complete Stage1-4 task definition.
MuJoCo Warp 3.11 requires zero geom margins for this model's collision pairs.
The controller uses position PD, but retains the official MJCF's torque caps.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
import mujoco

GO2_XML = (
    Path(__file__).resolve().parents[3]
    / "resources/robots/go2/go2_unitree_official.xml"
)
TRUNK_CONTACT_GEOM = "base_trunk_collision"
OFFICIAL_HIP_THIGH_EFFORT_NM = 23.7
OFFICIAL_CALF_EFFORT_NM = 45.43


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    # Name the existing base geoms in the adapter, preserving the frozen
    # official XML checksum and every collision/inertial parameter.
    base_collision_names = {
        mujoco.mjtGeom.mjGEOM_BOX: TRUNK_CONTACT_GEOM,
        mujoco.mjtGeom.mjGEOM_CYLINDER: "base_head_upper_collision",
        mujoco.mjtGeom.mjGEOM_SPHERE: "base_head_lower_collision",
    }
    for geom in spec.body("base_link").geoms:
        if geom.type in base_collision_names:
            geom.name = base_collision_names[geom.type]
    # Scene attachment does not propagate child-level <option> values; the
    # environment sets the desired elliptic cone and impratio explicitly.
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 1.0
    # The source asset contains torque motors, but mjlab's joint-position
    # action requires an explicit PD actuator. Do not leave both connected.
    for key in list(spec.keys):
        spec.delete(key)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for geom in spec.geoms:
        geom.margin = 0.0
    return spec


def get_go2_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        sort_actuators=True,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.42),
            joint_pos={
                "FL_hip_joint": -0.05,
                "FL_thigh_joint": 0.8,
                "FL_calf_joint": -1.5,
                "FR_hip_joint": 0.05,
                "FR_thigh_joint": 0.8,
                "FR_calf_joint": -1.5,
                "RL_hip_joint": -0.05,
                "RL_thigh_joint": 1.0,
                "RL_calf_joint": -1.5,
                "RR_hip_joint": 0.05,
                "RR_thigh_joint": 1.0,
                "RR_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*hip_joint", ".*thigh_joint"),
                    stiffness=20.0,
                    damping=0.5,
                    effort_limit=OFFICIAL_HIP_THIGH_EFFORT_NM,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*calf_joint",),
                    stiffness=20.0,
                    damping=0.5,
                    effort_limit=OFFICIAL_CALF_EFFORT_NM,
                ),
            ),
        ),
    )
