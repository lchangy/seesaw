"""Ten passive boards: true mass/inertia, fixed supports and limited Y hinges."""

from functools import partial
import math

import mujoco
from mjlab.entity import EntityCfg
from seesaw_policy.env.boarding.parameters import WIDTHS, HEIGHTS, ANGLES_DEG, DAMPING

ARENA_SPACING = (8.0, 5.0)


def arena_position(level, offset=(0.0, 0.0, 0.0)):
    return (
        offset[0] + level % 5 * ARENA_SPACING[0],
        offset[1] + level // 5 * ARENA_SPACING[1],
        offset[2],
    )


def board_spec(level, randomized=False, offset=(0.0, 0.0, 0.0)):
    width, height = WIDTHS[level], HEIGHTS[level]
    if randomized:
        width = 0.6
    angle = math.radians(ANGLES_DEG[level])
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    support = spec.worldbody.add_body(name="support", pos=arena_position(level, offset))
    support.add_geom(
        name="support_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0, 0, height / 2),
        size=(0.1, width * 0.3, height / 2),
        rgba=(0.25, 0.28, 0.32, 1),
        mass=0.0,
        group=0,
    )
    board = support.add_body(name="board", pos=(0, 0, height))
    board.add_joint(
        name="hinge",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=(0, 1, 0),
        limited=True,
        range=(-angle, angle),
        damping=DAMPING[level],
    )
    board.add_geom(
        name="board_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(1.8, width / 2, 0.02),
        mass=12.0,
        rgba=(0.72, 0.40 + level * 0.025, 0.12, 1),
        group=0,
        # Official feet have priority=1 and condim=6. Use board
        # parameters directly to avoid soft contact mixing and
        # rolling/torsional friction anchoring feet inside edges.
        priority=2,
        condim=3,
        solref=(0.002, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    )
    # A fixed support is welded to world, so the parent collision filter
    # does not suppress its overlap with the moving board at the pivot.
    # Exclude only this internal pair; feet/support and board/ground remain.
    spec.add_exclude(bodyname1="support", bodyname2="board")
    return spec


def board_entities(randomized=False, offset=(0.0, 0.0, 0.0)):
    return {
        f"seesaw_l{level}": EntityCfg(
            spec_fn=partial(board_spec, level, randomized, offset),
            init_state=EntityCfg.InitialStateCfg(
                joint_pos={"hinge": math.radians(ANGLES_DEG[level])},
                joint_vel={".*": 0.0},
            ),
        )
        for level in range(10)
    }


def arena_floor(offset):
    def spec_fn():
        spec = mujoco.MjSpec()
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(offset[0] + 16.0, offset[1] - 2.5, -0.1),
            size=(21.0, 16.0, 0.1),
            rgba=(0.3, 0.35, 0.3, 1),
            group=0,
        )
        return spec

    return EntityCfg(spec_fn=spec_fn)
