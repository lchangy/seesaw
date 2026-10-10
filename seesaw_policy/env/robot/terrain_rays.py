"""MuJoCo Warp ray patterns and Privileged critic terrain-data packing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

import numpy as np
import torch

from mjlab.sensor import RayCastSensorCfg
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.utils.lab_api.math import quat_apply
from .heightmap_geometry import heightmap_search_limits, footmap_search_limits


def yaw_local(vectors, quaternion):
    """World vectors to the shared gravity-aligned base-yaw frame."""
    w, x, y, z = quaternion.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    return torch.stack(
        (
            c * vectors[..., 0] + s * vectors[..., 1],
            -s * vectors[..., 0] + c * vectors[..., 1],
            vectors[..., 2],
        ),
        -1,
    )


@dataclass(kw_only=True)
class BaseYawFootRayCfg(RayCastSensorCfg):
    yaw_frame: ObjRef

    def build(self):
        return BaseYawFootRaySensor(self)


class BaseYawFootRaySensor(RayCastSensor):
    """Physical foot anchors with a common base-yaw grid, not foot-link yaw."""

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self._base_body_id = mj_model.body(self.cfg.yaw_frame.prefixed_name()).id

    def _compute_alignment_rotation(self, frame_mat):
        base_mat = self._data.xmat[:, self._base_body_id].view(-1, 3, 3)
        rotation = self._extract_yaw_rotation(base_mat)
        return rotation[:, None].expand(-1, self._num_frames, -1, -1).reshape(-1, 3, 3)


@dataclass(frozen=True)
class ExplicitRayPattern:
    """Ray pattern implementing MJLab's generate_rays protocol."""

    offsets: tuple[tuple[float, float, float], ...]
    directions: tuple[tuple[float, float, float], ...]

    def generate_rays(self, mj_model, device: str):
        del mj_model
        offsets = torch.tensor(self.offsets, dtype=torch.float32, device=device)
        directions = torch.tensor(self.directions, dtype=torch.float32, device=device)
        if (
            offsets.ndim != 2
            or offsets.shape[-1] != 3
            or directions.shape != offsets.shape
        ):
            raise ValueError(
                "explicit ray offsets/directions must both have shape [N, 3]"
            )
        norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        if torch.any(norms < 1e-8):
            raise ValueError("ray directions must be non-zero")
        return offsets, directions / norms


def _mesh_origins(xs, ys, z: float):
    xx, yy = np.meshgrid(
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        indexing="ij",
    )
    return np.stack((xx, yy, np.full_like(xx, z)), axis=-1).reshape(-1, 3)


def _repeat_direction(origin_count: int, direction):
    return np.broadcast_to(
        np.asarray(direction, dtype=np.float32), (origin_count, 3)
    ).copy()


def _spatial_privileged_pattern(spatial: dict[str, Any]):
    origins = []
    directions = []
    front_y, front_z = np.meshgrid(
        np.asarray(spatial["front_y"], dtype=np.float32),
        np.asarray(spatial["front_z"], dtype=np.float32),
        indexing="ij",
    )
    front = np.stack((np.zeros_like(front_y), front_y, front_z), axis=-1).reshape(-1, 3)
    origins.append(front)
    directions.append(_repeat_direction(len(front), (1.0, 0.0, 0.0)))

    up = _mesh_origins(spatial["up_x"], spatial["up_y"], float(spatial["up_z"]))
    origins.append(up)
    directions.append(_repeat_direction(len(up), (0.0, 0.0, 1.0)))

    side = []
    side_dirs = []
    for sign in (-1.0, 1.0):
        side_x, side_z = np.meshgrid(
            np.asarray(spatial["side_x"], dtype=np.float32),
            np.asarray(spatial["side_z"], dtype=np.float32),
            indexing="ij",
        )
        points = np.stack(
            (side_x, np.full_like(side_x, sign * float(spatial["side_y"])), side_z),
            axis=-1,
        ).reshape(-1, 3)
        side.append(points)
        side_dirs.append(_repeat_direction(len(points), (0.0, sign, 0.0)))
    origins.extend(side)
    directions.extend(side_dirs)
    origin_array = np.concatenate(origins, axis=0)
    direction_array = np.concatenate(directions, axis=0)
    limits = np.concatenate(
        (
            np.full(len(front), float(spatial["front_range"]), dtype=np.float32),
            np.full(
                len(up) + sum(len(v) for v in side),
                float(spatial["near_range"]),
                dtype=np.float32,
            ),
        )
    )
    if len(origin_array) != 130:
        raise ValueError(
            f"Privileged privileged spatial pattern must have 130 rays, got {len(origin_array)}"
        )
    return origin_array, direction_array, limits


def make_terrain_ray_sensors(recipe: Any) -> tuple[RayCastSensorCfg, ...]:
    """Create GPU ray sensors for terrain maps, spatial context, and body probes."""
    env_cfg = recipe.env_config
    geometry = recipe.train_config["policy"]["critic_terrain_schema"]
    spatial = env_cfg["geometry"]
    base = ObjRef(type="body", name="base_link", entity="robot")

    # The first 1974 rays are ordered as front/back/left/right/up/down and are
    # consumed by the learned Privileged geometry encoder.  The following 882
    # downward rays are the critic's ground map, then 130 legacy spatial rays.
    origins = np.asarray(geometry["ray_origins"], dtype=np.float32)
    directions = np.asarray(geometry["ray_directions"], dtype=np.float32)
    limits = np.asarray(geometry["ray_limits"], dtype=np.float32)
    if (
        origins.shape != (1974, 3)
        or directions.shape != origins.shape
        or limits.shape != (1974,)
    ):
        raise ValueError("Privileged critic ray schema changed from 1974 rays")

    height_ceiling, height_range = heightmap_search_limits(geometry)
    ground_origins = _mesh_origins(
        geometry["height_x"], geometry["height_y"], height_ceiling
    )
    ground_directions = _repeat_direction(len(ground_origins), (0.0, 0.0, -1.0))
    aux_origins, aux_directions, aux_limits = _spatial_privileged_pattern(spatial)
    main_pattern = ExplicitRayPattern(
        offsets=tuple(
            map(
                tuple,
                np.concatenate((origins, ground_origins, aux_origins), axis=0).tolist(),
            )
        ),
        directions=tuple(
            map(
                tuple,
                np.concatenate(
                    (directions, ground_directions, aux_directions), axis=0
                ).tolist(),
            )
        ),
    )
    main_sensor = RayCastSensorCfg(
        name="spatial_geometry_rays",
        frame=base,
        pattern=main_pattern,
        ray_alignment="yaw",
        max_distance=max(float(limits.max()), height_range, float(aux_limits.max())),
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )

    foot_names = tuple(geometry["foot_names"])
    foot_ceiling, foot_range = footmap_search_limits(geometry)
    foot_origins = _mesh_origins(
        geometry["foot_axis"], geometry["foot_axis"], foot_ceiling
    )
    foot_sensor = BaseYawFootRayCfg(
        name="spatial_foot_rays",
        yaw_frame=base,
        frame=tuple(
            ObjRef(type="body", name=name, entity="robot") for name in foot_names
        ),
        pattern=ExplicitRayPattern(
            offsets=tuple(map(tuple, foot_origins.tolist())),
            directions=tuple(
                map(
                    tuple,
                    _repeat_direction(len(foot_origins), (0.0, 0.0, -1.0)).tolist(),
                )
            ),
        ),
        ray_alignment="yaw",
        max_distance=foot_range,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )

    # The schema stores 8 base-link probes and two probes on each thigh/calf.
    six_directions = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=np.float32,
    )
    probes = geometry["probes"]
    base_probes = [p for p in probes if p["link"] == "base"]
    base_offsets = np.repeat(
        np.asarray([p["local_position"] for p in base_probes], dtype=np.float32),
        6,
        axis=0,
    )
    base_dirs = np.tile(six_directions, (len(base_probes), 1))
    base_probe_sensor = RayCastSensorCfg(
        name="spatial_base_probe_rays",
        frame=base,
        pattern=ExplicitRayPattern(
            tuple(map(tuple, base_offsets.tolist())),
            tuple(map(tuple, base_dirs.tolist())),
        ),
        ray_alignment="base",
        max_distance=float(geometry["search"]["body_range"]),
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    leg_names = tuple(dict.fromkeys(p["link"] for p in probes if p["link"] != "base"))
    offsets_by_link = {
        name: [p["local_position"] for p in probes if p["link"] == name]
        for name in leg_names
    }
    if any(len(values) != 2 for values in offsets_by_link.values()):
        raise ValueError("Expected two Privileged surface probes per thigh/calf link")
    leg_offsets = np.repeat(
        np.asarray(offsets_by_link[leg_names[0]], dtype=np.float32), 6, axis=0
    )
    leg_dirs = np.tile(six_directions, (len(offsets_by_link[leg_names[0]]), 1))
    leg_probe_sensor = RayCastSensorCfg(
        name="spatial_leg_probe_rays",
        frame=tuple(
            ObjRef(type="body", name=name, entity="robot") for name in leg_names
        ),
        pattern=ExplicitRayPattern(
            tuple(map(tuple, leg_offsets.tolist())),
            tuple(map(tuple, leg_dirs.tolist())),
        ),
        ray_alignment="base",
        max_distance=float(geometry["search"]["body_range"]),
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    return main_sensor, foot_sensor, base_probe_sensor, leg_probe_sensor


class TerrainRayReader:
    """Pack Privileged terrain tensors from live MuJoCo Warp ray sensor buffers."""

    def __init__(self, env, recipe: Any):
        self.env = env
        self.device = torch.device(env.device)
        self.recipe = recipe
        self.geometry = recipe.train_config["policy"]["critic_terrain_schema"]
        self.height_ceiling, self.height_range = heightmap_search_limits(self.geometry)
        self.foot_ceiling, self.foot_range = footmap_search_limits(self.geometry)
        self.spatial = recipe.env_config["geometry"]
        _, _, privileged_limits = _spatial_privileged_pattern(self.spatial)
        self.privileged_limits = torch.as_tensor(
            privileged_limits, dtype=torch.float32, device=env.device
        )
        self.main_slices = {
            "spatial": slice(0, 1974),
            "heightmap": slice(
                1974,
                1974 + len(self.geometry["height_x"]) * len(self.geometry["height_y"]),
            ),
            "privileged": slice(
                1974 + len(self.geometry["height_x"]) * len(self.geometry["height_y"]),
                None,
            ),
        }
        self.body_names = tuple(env.scene["robot"].body_names)
        body_index = {name: index for index, name in enumerate(self.body_names)}
        self.foot_ids = torch.tensor(
            [body_index[name] for name in self.geometry["foot_names"]],
            device=env.device,
            dtype=torch.long,
        )
        self._decoded_key = None
        self._decoded_refs = None
        self._decoded_data = None
        self._profile_read_limit = 0
        self._profile_read_count = 0

    def start_profiling(self, read_count: int) -> None:
        self._profile_read_limit = max(int(read_count), 0)
        self._profile_read_count = 0

    @torch.no_grad()
    def _nearest_probe_vectors(
        self, sensor, local_offsets: np.ndarray, probes_per_frame: int
    ):
        rays = sensor.data
        num_envs = self.env.num_envs
        num_frames = rays.frame_pos_w.shape[1]
        directions = 6
        local = torch.as_tensor(
            local_offsets, dtype=torch.float32, device=self.env.device
        )
        local = local[:, None, :].expand(-1, directions, -1)
        frame_pos = rays.frame_pos_w
        frame_quat = rays.frame_quat_w
        repeated_quat = (
            frame_quat[:, :, None, None, :]
            .expand(num_envs, num_frames, probes_per_frame, directions, 4)
            .reshape(-1, 4)
        )
        repeated_local = (
            local[None, None].expand(num_envs, num_frames, -1, -1, -1).reshape(-1, 3)
        )
        origin = frame_pos[:, :, None, None, :] + quat_apply(
            repeated_quat, repeated_local
        ).reshape(num_envs, num_frames, probes_per_frame, directions, 3)
        hit_pos = rays.hit_pos_w.reshape(
            num_envs, num_frames, probes_per_frame, directions, 3
        )
        distance = rays.distances.reshape(
            num_envs, num_frames, probes_per_frame, directions
        )
        vector = hit_pos - origin
        valid = (distance >= 0.0) & torch.isfinite(distance)
        valid &= torch.isfinite(vector).all(-1)
        score = torch.where(valid, distance, torch.full_like(distance, float("inf")))
        nearest = score.argmin(dim=-1)
        gather = nearest[..., None, None].expand(-1, -1, -1, 1, 3)
        selected = vector.gather(-2, gather).squeeze(-2)
        found = valid.any(dim=-1)
        return (
            torch.where(found[..., None], selected, torch.zeros_like(selected)),
            found,
        )

    @torch.no_grad()
    def read_heightmap(self):
        """Read current world-vertical ground rays without packing all geometry."""
        main = self.env.scene["spatial_geometry_rays"].data
        height_slice = self.main_slices["heightmap"]
        distance = main.distances[:, height_slice]
        relative_z = (
            main.hit_pos_w[:, height_slice, 2]
            - self.env.scene["robot"].data.root_link_pos_w[:, 2:3]
        )
        valid = (distance >= 0.0) & (distance <= self.height_range)
        valid &= torch.isfinite(relative_z)
        shape = (
            self.env.num_envs,
            len(self.geometry["height_x"]),
            len(self.geometry["height_y"]),
        )
        return {
            "height": torch.where(valid, relative_z, 0.0).reshape(shape),
            "valid": valid.reshape(shape),
        }

    @torch.no_grad()
    def read(self, *, purpose: str = "state", pack: bool = True):
        env = self.env
        profile = self._profile_read_count < self._profile_read_limit
        if profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        profile_started = time.perf_counter() if profile else None
        state = env.scene["robot"].data
        n = env.num_envs
        main = env.scene["spatial_geometry_rays"].data
        # Sensor data objects are renewed on update/sense/reset. Retain references
        # so Python IDs cannot be recycled while the decoded result is cached.
        refs = (
            main,
            env.scene["spatial_foot_rays"].data,
            env.scene["spatial_base_probe_rays"].data,
            env.scene["spatial_leg_probe_rays"].data,
        )
        key = (
            tuple(id(value) for value in refs),
            env.episode_length_buf._version,
            env.sim.data.qpos._version,
        )
        if not pack and key == self._decoded_key and not profile:
            return self._decoded_data, None
        height_shape = (len(self.geometry["height_x"]), len(self.geometry["height_y"]))
        heightmap = self.read_heightmap()

        foot = env.scene["spatial_foot_rays"].data
        foot_z = foot.frame_pos_w[:, :, 2]
        foot_dist = foot.distances.reshape(n, 4, -1)
        foot_valid = (foot_dist >= 0.0) & (foot_dist <= self.foot_range)
        fresh = env.episode_length_buf > 0
        foot_valid &= fresh[:, None, None]
        foot_hit_height = foot.hit_pos_w[..., 2].reshape(n, 4, -1) - foot_z[:, :, None]
        foot_valid &= torch.isfinite(foot_hit_height)
        foot_size = len(self.geometry["foot_axis"])
        footmap = {
            "height": torch.where(foot_valid, foot_hit_height, 0.0).reshape(
                n, 4, foot_size, foot_size
            ),
            "valid": foot_valid.reshape(n, 4, foot_size, foot_size),
        }

        ray_slice = self.main_slices["spatial"]
        ray_dist = main.distances[:, ray_slice]
        ray_limits = torch.as_tensor(
            self.geometry["ray_limits"], dtype=torch.float32, device=env.device
        )
        base_quaternion = state.root_link_quat_w
        ray_normals = yaw_local(main.normals_w[:, ray_slice], base_quaternion)
        ray_valid = (ray_dist >= 0.0) & (ray_dist <= ray_limits[None])
        ray_valid &= torch.isfinite(ray_normals).all(-1)
        spatial = {
            "distance": torch.where(ray_valid, ray_dist, ray_limits[None]),
            "valid": ray_valid,
            "normal": torch.where(ray_valid[..., None], ray_normals, 0.0),
        }
        privileged_dist = main.distances[:, self.main_slices["privileged"]]
        privileged_valid = (privileged_dist >= 0.0) & (
            privileged_dist <= self.privileged_limits[None]
        )
        spatial_privileged = torch.cat(
            (
                torch.where(
                    privileged_valid,
                    privileged_dist / self.privileged_limits[None],
                    torch.ones_like(privileged_dist),
                ),
                privileged_valid.to(torch.float32),
            ),
            dim=1,
        )

        base_probe_sensor = env.scene["spatial_base_probe_rays"]
        base_offsets = np.asarray(
            [
                p["local_position"]
                for p in self.geometry["probes"]
                if p["link"] == "base"
            ],
            dtype=np.float32,
        )
        base_vector, base_valid = self._nearest_probe_vectors(
            base_probe_sensor, base_offsets, len(base_offsets)
        )
        leg_probe_sensor = env.scene["spatial_leg_probe_rays"]
        leg_names = tuple(
            dict.fromkeys(
                p["link"] for p in self.geometry["probes"] if p["link"] != "base"
            )
        )
        leg_offsets = np.asarray(
            [
                p["local_position"]
                for p in self.geometry["probes"]
                if p["link"] == leg_names[0]
            ],
            dtype=np.float32,
        )
        leg_vector, leg_valid = self._nearest_probe_vectors(
            leg_probe_sensor, leg_offsets, len(leg_offsets)
        )
        # Source schema order is base corners, then FL/FR/RL/RR thigh/calf.
        # The multi-frame ray sensor preserves the configured body-list order.
        body_vector = torch.cat(
            (base_vector.flatten(1), leg_vector.flatten(1)), dim=1
        ).reshape(n, 24, 3)
        body_vector = yaw_local(body_vector, base_quaternion)
        body_valid = torch.cat(
            (base_valid.flatten(1), leg_valid.flatten(1)), dim=1
        ).reshape(n, 24)
        body_valid &= fresh[:, None]
        body = {
            "vector": torch.where(body_valid[..., None], body_vector, 0.0),
            "valid": body_valid,
        }
        data = {
            "heightmap": heightmap,
            "footmap": footmap,
            "spatial": spatial,
            "body": body,
            # The learned geometry packer intentionally ignores this extra
            # field; the 1510-D shared-privileged critic consumes it directly.
            "spatial_privileged": spatial_privileged,
        }
        packed = None
        if pack:
            from seesaw_policy.terrain_data import pack_terrain_data

            packed = pack_terrain_data(data)
            expected = (
                height_shape[0] * height_shape[1] * 2
                + 4 * foot_size * foot_size * 2
                + 1974 * 5
                + 24 * 4
            )
            if packed.shape != (n, expected) or not torch.isfinite(packed).all():
                raise AssertionError(
                    f"Packed Privileged terrain shape/finite check failed: {tuple(packed.shape)}"
                )
        if (
            spatial_privileged.shape != (n, 260)
            or not torch.isfinite(spatial_privileged).all()
        ):
            raise AssertionError(
                "Packed privileged spatial rays must be finite [N, 260]"
            )
        if profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            runtime = getattr(env, "spatial_runtime", None)
            print(
                "MJLAB_RAY_READ_TIMING "
                + json.dumps(
                    {
                        "pool": getattr(runtime, "pool", "unknown"),
                        "purpose": purpose,
                        "packed": pack,
                        "call": self._profile_read_count,
                        "envs": int(n),
                        "seconds": round(time.perf_counter() - profile_started, 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            self._profile_read_count += 1
        self._decoded_key, self._decoded_refs, self._decoded_data = key, refs, data
        return data, packed
