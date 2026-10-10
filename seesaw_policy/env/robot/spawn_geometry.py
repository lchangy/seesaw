"""Collision bounds for penetration-free robot resets."""

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_mul
from .asset import get_spec


class Go2SpawnClearance:

    def __init__(self, joint_names, device):
        model = get_spec().compile()
        base = model.body("base_link").id
        body_ids = list(range(base, model.nbody))
        index = {body: i for i, body in enumerate(body_ids)}
        self.num_bodies = len(body_ids)
        array = lambda x: torch.as_tensor(
            np.asarray(x).copy(), dtype=torch.float32, device=device
        )
        long = lambda x: torch.tensor(x, dtype=torch.long, device=device)
        self.parents = long(
            [0] + [index[int(model.body_parentid[b])] for b in body_ids[1:]]
        )
        self.body_pos = array(model.body_pos[body_ids])
        self.body_quat = array(model.body_quat[body_ids])
        self.body_joint = long([0] * len(body_ids))
        self.joint_axis = array(np.zeros((len(body_ids), 3)))
        depth = [0] * len(body_ids)
        for i, body in enumerate(body_ids[1:], 1):
            depth[i] = depth[int(self.parents[i])] + 1
            count = int(model.body_jntnum[body])
            if count > 1:
                raise ValueError("Spawn FK expects at most one hinge per non-root body")
            if count:
                joint = int(model.body_jntadr[body])
                if model.jnt_type[
                    joint
                ] != mujoco.mjtJoint.mjJNT_HINGE or not np.allclose(
                    model.jnt_pos[joint], 0
                ):
                    raise ValueError("Spawn FK requires a hinge at the body origin")
                name = model.joint(joint).name
                self.body_joint[i] = joint_names.index(name)
                self.joint_axis[i] = array(model.jnt_axis[joint])
        self.depth_groups = [
            long([i for i, d in enumerate(depth) if d == level])
            for level in range(1, max(depth) + 1)
        ]
        geom_ids = [
            i
            for i in range(model.ngeom)
            if int(model.geom_bodyid[i]) in index
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
        self.geom_ids = geom_ids
        self.geom_bodies = long([index[int(model.geom_bodyid[g])] for g in geom_ids])
        self.geom_pos = array(model.geom_pos[geom_ids])
        self.geom_quat = array(model.geom_quat[geom_ids])
        self.geom_size = array(model.geom_size[geom_ids])
        types = model.geom_type[geom_ids]
        self.type_masks = {
            name: torch.as_tensor(types == kind, device=device)
            for name, kind in (
                ("sphere", mujoco.mjtGeom.mjGEOM_SPHERE),
                ("box", mujoco.mjtGeom.mjGEOM_BOX),
                ("cylinder", mujoco.mjtGeom.mjGEOM_CYLINDER),
                ("capsule", mujoco.mjtGeom.mjGEOM_CAPSULE),
            )
        }
        if not torch.stack(list(self.type_masks.values())).any(0).all():
            raise ValueError("Unsupported collision shape in Go2 spawn FK")
        joint_ids = [model.joint(name).id for name in joint_names]
        self.joint_limits = array(model.jnt_range[joint_ids])

    def collision_bounds(self, joints, root_quat):
        """World-axis collision AABB relative to root translation, per row."""
        n = joints.shape[0]
        positions = joints.new_zeros((n, self.num_bodies, 3))
        quats = joints.new_zeros((n, self.num_bodies, 4))
        quats[:, 0] = root_quat
        for ids in self.depth_groups:
            parent = self.parents[ids]
            parent_quat = quats[:, parent]
            positions[:, ids] = positions[:, parent] + quat_apply(
                parent_quat, self.body_pos[ids].expand(n, -1, -1)
            )
            half_angle = joints[:, self.body_joint[ids]] * 0.5
            rotation = torch.cat(
                (
                    half_angle.cos()[..., None],
                    half_angle.sin()[..., None] * self.joint_axis[ids],
                ),
                -1,
            )
            fixed = self.joint_axis[ids].square().sum(-1) == 0
            rotation[:, fixed] = rotation.new_tensor((1.0, 0.0, 0.0, 0.0))
            quats[:, ids] = quat_mul(
                quat_mul(parent_quat, self.body_quat[ids].expand(n, -1, -1)), rotation
            )
        centers = positions[:, self.geom_bodies] + quat_apply(
            quats[:, self.geom_bodies], self.geom_pos.expand(n, -1, -1)
        )
        rotations = matrix_from_quat(
            quat_mul(quats[:, self.geom_bodies], self.geom_quat.expand(n, -1, -1))
        )
        extents = torch.zeros_like(centers)
        for kind, mask in self.type_masks.items():
            size = self.geom_size[mask]
            if kind == "sphere":
                extents[:, mask] = size[:, :1]
            elif kind == "box":
                extents[:, mask] = (
                    rotations[:, mask].abs() * size[None, :, None, :]
                ).sum(-1)
            else:
                axis = rotations[:, mask, :, 2].abs()
                radial = (
                    (1 - axis.square()).clamp_min(0).sqrt()
                    if kind == "cylinder"
                    else torch.ones_like(axis)
                )
                extents[:, mask] = (
                    size[None, :, :1] * radial + size[None, :, 1:2] * axis
                )
        return ((centers - extents).amin(1), (centers + extents).amax(1))

    def place(self, root, joints, origins, *, half_extent, clearance=0.015):
        joints = joints.clamp(self.joint_limits[:, 0], self.joint_limits[:, 1])
        low, high = self.collision_bounds(joints, root[:, 3:7])
        lower_xy = origins[:, :2] - half_extent - low[:, :2]
        upper_xy = origins[:, :2] + half_extent - high[:, :2]
        if bool((lower_xy > upper_xy).any()):
            raise ValueError(
                "Sampled Go2 pose does not fit inside the stair spawn platform"
            )
        root = root.clone()
        root[:, :2] = root[:, :2].clamp(lower_xy, upper_xy)
        root[:, 2] = torch.maximum(root[:, 2], origins[:, 2] + clearance - low[:, 2])
        return (root, joints)
