"""Batched body-mounted pinhole depth from the live MJLab scene."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CameraPattern:
    config: object

    def generate_rays(self, mj_model, device):
        del mj_model
        cfg = self.config
        # Optical coordinates: x forward, y left, z up. Pixel centres in
        # row-major order; rotate positive pitch down about body y.
        focal = cfg.camera_width / (2 * math.tan(math.radians(cfg.camera_horizontal_fov) / 2))
        row, col = torch.meshgrid(
            torch.arange(cfg.camera_height, device=device),
            torch.arange(cfg.camera_width, device=device), indexing='ij',
        )
        rays = torch.stack((torch.ones_like(row),
                            (cfg.camera_width / 2 - col - .5) / focal,
                            (cfg.camera_height / 2 - row - .5) / focal), -1).float().reshape(-1, 3)
        pitch = math.radians(cfg.camera_pitch)
        rotation = rays.new_tensor(((math.cos(pitch), 0, math.sin(pitch)),
                                    (0, 1, 0), (-math.sin(pitch), 0, math.cos(pitch))))
        rays = torch.nn.functional.normalize(rays, dim=-1) @ rotation.T
        origins = rays.new_tensor(cfg.camera_position).expand_as(rays).contiguous()
        return origins, rays.contiguous()


def mount_rotation(angles):
    """Body-local Rz(yaw) Ry(pitch) Rx(roll); positive pitch points down."""
    roll, pitch, yaw = torch.deg2rad(angles).unbind(-1)
    cr, cp, cy = roll.cos(), pitch.cos(), yaw.cos()
    sr, sp, sy = roll.sin(), pitch.sin(), yaw.sin()
    return torch.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                        sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                        -sp, cp*sr, cp*cr), -1).reshape(-1, 3, 3)


class CameraMountDR:
    """One independent installation per episode, never frame-to-frame noise."""
    def __init__(self, config, count, device):
        self.config, self.enabled = config, config.camera_mount_dr
        self.active = self.enabled
        self.position = torch.tensor(config.camera_position, device=device).repeat(count, 1)
        self.angles = self.position.new_tensor((0., config.camera_pitch, 0.)).repeat(count, 1)
        self.rotation = mount_rotation(self.angles)
        self.strength = self.position.new_zeros(count)

    @torch.no_grad()
    def reset_envs(self, mask):
        ids = mask.nonzero(as_tuple=False).flatten()
        if not len(ids):
            return ids
        cfg = self.config
        position = self.position.new_tensor(cfg.camera_position).expand(len(ids), 3)
        angles = self.angles.new_tensor((0., cfg.camera_pitch, 0.)).expand(len(ids), 3)
        strength = self.strength.new_zeros(len(ids))
        if self.enabled:
            self.active = True
            draw = torch.rand(len(ids), device=self.position.device)
            strength = torch.where(draw < cfg.camera_mount_clean_probability, 0.,
                torch.where(draw < cfg.camera_mount_clean_probability+cfg.camera_mount_light_probability,
                            cfg.camera_mount_light_strength, 1.))
            error = torch.rand(len(ids), 6, device=self.position.device)*2-1
            position = position + error[:, :3]*position.new_tensor(cfg.camera_mount_position_range)*strength[:, None]
            angles = angles + error[:, 3:]*angles.new_tensor(cfg.camera_mount_rotation_range)*strength[:, None]
        self.position[ids], self.angles[ids], self.strength[ids] = position, angles, strength
        self.rotation[ids] = mount_rotation(angles)
        return ids

    def world_rays(self, frame_position, frame_rotation, optical_rays, env_slice):
        rotation = frame_rotation @ self.rotation[env_slice]
        origin = frame_position + torch.einsum('bij,bj->bi', frame_rotation, self.position[env_slice])
        rays = torch.einsum('bij,nj->bni', rotation, optical_rays)
        return origin[:, None].expand_as(rays), rays


def normalize_depth(distances, config):
    distances = torch.where((distances >= 0) & torch.isfinite(distances),
                            distances, config.camera_far)
    # Convert optical depth to metres, then
    # maps near to -0.5 and far/miss to +0.5.
    image = (distances.clamp(config.camera_near, config.camera_far)
             - config.camera_near) / (config.camera_far - config.camera_near) - .5
    return image.reshape(-1, config.camera_height, config.camera_width, 1)


@torch.no_grad()
def depth_augmentation_strength(batch, device, config):
    draw = torch.rand(batch, 1, 1, 1, device=device)
    return torch.where(draw < config.depth_clean_probability, 0.,
        torch.where(draw < config.depth_clean_probability+config.depth_light_probability,
                    config.depth_light_strength, 1.))


@torch.no_grad()
def augment_depth(image, config, *, severity=None, strength=None):
    """Training mix, or viewer-only 0..1 severity (1 = sensor blackout).

    Manual 0.5 matches full training intensity; above it increasingly drops
    pixels. Missing depth stays +0.5, never the false near obstacle of -0.5.
    """
    if severity is not None:
        if not 0 <= severity <= 1:
            raise ValueError('Camera image severity must be in [0, 1]')
        if severity == 0:
            return image
        if severity == 1:
            return torch.full_like(image, .5)
    elif not config.depth_augmentation:
        return image
    batch, height, width, _ = image.shape
    device = image.device
    original = image
    if severity is not None:
        strength = image.new_full((batch, 1, 1, 1), min(2*severity, 1.))
    elif strength is None:
        strength = depth_augmentation_strength(batch, device, config)
    image = image.clone()
    depth = (original + .5) * (config.camera_far - config.camera_near) + config.camera_near
    near_invalid = (depth < .15) & (strength > 0)
    valid = image < .5
    if config.depth_noise_std:
        noise = torch.randn_like(image) * (config.depth_noise_std / (config.camera_far - config.camera_near))*strength
        image = torch.where(valid, image + noise, image).clamp(-.5, .5)
    if config.depth_blur_probability:
        sigma = torch.empty((), device=device).uniform_(*config.depth_blur_sigma)
        coordinates = torch.arange(-5, 6, device=device, dtype=image.dtype)
        kernel = torch.exp(-coordinates.square() / (2 * sigma.square()))
        kernel /= kernel.sum()
        pixels = image.permute(0, 3, 1, 2)
        blurred = F.conv2d(F.pad(pixels, (5, 5, 0, 0), mode='replicate'), kernel.view(1, 1, 1, 11))
        blurred = F.conv2d(F.pad(blurred, (0, 0, 5, 5), mode='replicate'), kernel.view(1, 1, 11, 1))
        mask = torch.rand(batch, 1, 1, 1, device=device) < config.depth_blur_probability*strength
        image = torch.where(mask, image.lerp(blurred.permute(0, 2, 3, 1), strength), image)
    if config.depth_occlusion_probability:
        area = torch.empty(batch, 1, 1, 1, device=device).uniform_(*config.depth_occlusion_area)
        size = (area*strength).sqrt()
        top = torch.rand_like(size) * (1 - size)
        left = torch.rand_like(size) * (1 - size)
        y = (torch.arange(height, device=device) + .5).view(1, height, 1, 1) / height
        x = (torch.arange(width, device=device) + .5).view(1, 1, width, 1) / width
        mask = (y >= top) & (y < top + size) & (x >= left) & (x < left + size)
        mask &= torch.rand_like(size) < config.depth_occlusion_probability*strength
        image.masked_fill_(mask, .5)
    if config.depth_pixel_dropout:
        image.masked_fill_(torch.rand_like(image) < config.depth_pixel_dropout*strength, .5)
    if config.depth_frame_dropout:
        image.masked_fill_(torch.rand(batch, 1, 1, 1, device=device) < config.depth_frame_dropout*strength, .5)
    if severity is not None and severity > .5:
        image.masked_fill_(torch.rand_like(image) < 2*severity-1, .5)
    image.masked_fill_(near_invalid, .5)
    return torch.where(strength == 0, original, image)


class MJlabDepthCamera:
    def __init__(self, env, config):
        from mjlab.sensor import RayCastSensorCfg
        from mjlab.sensor.builtin_sensor import ObjRef
        from mjlab.sensor.sensor_context import SensorContext

        self.config = config
        # Runtime-only playback override. None leaves training exactly intact.
        self.augmentation_severity = None
        self.augmentation_strength = None  # Trainer's episode-constant group.
        self.augmentation_level = None  # Explicit train-only control profile.
        self.last_augmentation_severity = None
        self.mounts = CameraMountDR(config, env.num_envs, env.device)
        from .image_dr import StructuredDepthDR
        self.image_dr = StructuredDepthDR(env.num_envs, config.camera_height, config.camera_width,
                                         env.device, near=config.camera_near, far=config.camera_far)
        self.image_dr_time = 0.
        self.items = []
        from dataclasses import replace
        _, directions = CameraPattern(replace(config, camera_pitch=0)).generate_rays(None, str(env.device))
        self.optical_cosine = directions[:, 0]
        self.optical_rays = directions
        for lab_env in env.lab_envs:
            sim = lab_env.sim
            sensor = RayCastSensorCfg(
                name='seesaw_depth',
                frame=ObjRef(type='body', name='base_link', entity='robot'),
                pattern=CameraPattern(config), ray_alignment='base',
                max_distance=config.camera_far / float(self.optical_cosine.min()), exclude_parent_body=True,
                include_geom_groups=(0,),
            ).build()
            sensor.initialize(sim.mj_model, sim.model, sim.data, str(env.device))
            # A separate context needs no changes to the scene or its
            # captured graphs. It reads the same live Warp model/data (incl. DR).
            context = SensorContext(sim.mj_model, sim.model, sim.data,
                                    [], [sensor], str(env.device))
            self.items.append((sim, sensor, context))

    def reset_envs(self, mask):
        return self.mounts.reset_envs(mask)

    def set_image_dr_time(self, time, first):
        self.image_dr_time = time
        self.image_dr.reset_envs(first)

    def _apply_mounts(self, sensor, env_slice):
        import warp as wp
        # SensorContext prepares nominal shared rays. Replace their Warp input
        # buffers with per-world camera poses before the ray kernel executes.
        origins, rays = self.mounts.world_rays(sensor._cached_frame_pos[:, 0],
            sensor._cached_frame_mat[:, 0], self.optical_rays, env_slice)
        wp.to_torch(sensor._ray_pnt).view_as(origins).copy_(origins)
        wp.to_torch(sensor._ray_vec).view_as(rays).copy_(rays)
        sensor._cached_world_origins, sensor._cached_world_rays = origins, rays

    @torch.no_grad()
    def capture(self):
        import mujoco_warp as mjwarp
        import warp as wp

        images, clean_images = [], []
        severity = self.augmentation_severity
        offset = 0
        for sim, sensor, context in self.items:
            context.prepare()
            count = sensor._cached_frame_pos.shape[0]
            if self.mounts.active:
                self._apply_mounts(sensor, slice(offset, offset+count))
            strength = (None if self.augmentation_strength is None else
                        self.augmentation_strength[offset:offset+count])
            ids = torch.arange(offset, offset+count, device=self.mounts.position.device)
            offset += count
            with wp.ScopedDevice(sim.wp_device):
                mjwarp.refit_bvh(sim.wp_model, sim.wp_data, context.render_context)
                sensor.raycast_kernel(context.render_context)
            context.finalize()
            sensor.update(0.0)
            distances = sensor.data.distances
            # Rays report range; the policy consumes optical-axis depth.
            depth = torch.where(distances >= 0, distances * self.optical_cosine, distances)
            clean = normalize_depth(depth, self.config)
            clean_images.append(clean.clone())
            if self.config.depth_structured_dr and self.config.depth_augmentation and severity is None:
                levels = (torch.full((count,), 3, device=clean.device, dtype=torch.long)
                          if strength is None else torch.where(strength.flatten() == 0, 0,
                              torch.where(strength.flatten() < .4, 1,
                                  torch.where(strength.flatten() < .8, 2, 3))))
                if self.augmentation_level is not None:
                    levels.fill_(self.augmentation_level)
                images.append(self.image_dr.apply(clean, ids, levels, self.image_dr_time))
            else:
                images.append(augment_depth(clean, self.config, severity=severity, strength=strength))
        self.last_augmentation_severity = severity
        self.clean_image = torch.cat(clean_images)
        return torch.cat(images)
