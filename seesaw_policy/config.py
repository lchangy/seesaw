from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass(frozen=True)
class SensorConfig:
    camera_height: int = 32
    camera_width: int = 48
    # Cropped D435 geometry. Positive pitch points down.
    camera_position: tuple = (0.333, 0.0, 0.058)
    camera_pitch: float = 5.0
    camera_horizontal_fov: float = 73.71258352681936
    camera_near: float = 0.0
    camera_far: float = 3.0
    camera_interval: int = 5
    # Episode-constant camera mount errors.
    camera_mount_dr: bool = False
    camera_mount_position_range: tuple = (.02, .01, .01)  # XYZ, metres.
    camera_mount_rotation_range: tuple = (2., 5., 3.)  # Roll/pitch/yaw, degrees.
    camera_mount_clean_probability: float = .30
    camera_mount_light_probability: float = .30
    camera_mount_light_strength: float = .50
    depth_augmentation: bool = True
    depth_structured_dr: bool = False
    depth_clean_probability: float = .30
    depth_light_probability: float = .30
    depth_light_strength: float = .50
    depth_noise_std: float = 0.015  # Optical depth noise in metres.
    depth_pixel_dropout: float = 0.05
    depth_frame_dropout: float = 0.08
    depth_blur_probability: float = 0.55
    depth_blur_sigma: tuple = (1.0, 2.2)
    depth_occlusion_probability: float = 0.25
    depth_occlusion_area: tuple = (0.10, 0.35)

    def to_dict(self):
        return asdict(self)


def memory_defaults():
    values = json.loads((Path(__file__).parent/"vendor/world_model.json").read_text())
    values.update(dyn_deter=256, dyn_hidden=256, dyn_stoch=16, dyn_discrete=32)
    values["encoder"].update(mlp_units=256, mlp_layers=3)
    return values
