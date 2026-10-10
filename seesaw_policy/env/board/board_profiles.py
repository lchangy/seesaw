"""Base board DR and an explicitly enabled seesaw 3.5 continuation ramp."""

from copy import deepcopy

BASE_BOARD_DR = dict(
    enabled=True,
    profile="base",
    mass_range_kg=[11.0, 13.0],
    damping_range=[0.6, 0.9],
    friction_range=[0.8, 1.0],
    width_range_m=[0.50, 0.60],
)
ENHANCED_BOARD_DR = dict(
    enabled=True,
    profile="seesaw3.5",
    mass_range_kg=[8.0, 16.0],
    damping_range=[0.05, 1.5],
    friction_range=[0.4, 1.2],
    width_range_m=[0.25, 0.60],
)
RANGE_KEYS = ("mass_range_kg", "damping_range", "friction_range", "width_range_m")


def board_profile(profile, start_iteration, ramp_iterations=2000):
    if profile == "base":
        return deepcopy(BASE_BOARD_DR)
    if profile != "seesaw3.5" or ramp_iterations < 1:
        raise ValueError("Use base or seesaw3.5 with a positive ramp")
    config = deepcopy(ENHANCED_BOARD_DR)
    config["seesaw3_5"] = dict(
        start_iteration=int(start_iteration),
        ramp_iterations=int(ramp_iterations),
        base_ranges=deepcopy(BASE_BOARD_DR),
        base_fraction=0.3,
    )
    return config


def board_dr_progress(config, iteration):
    ramp = config.get("seesaw3_5")
    if ramp is None:
        return 0.0
    return min(
        1.0, max(0.0, (iteration - ramp["start_iteration"]) / ramp["ramp_iterations"])
    )


def board_length_progress(config, iteration):
    ramp = config.get("length_ramp")
    if ramp is None:
        return float("length_range_m" in config)
    return min(
        1.0, max(0.0, (iteration - ramp["start_iteration"]) / ramp["ramp_iterations"])
    )


def effective_board_config(config, iteration):
    ramp = config.get("seesaw3_5")
    if ramp is None and "length_ramp" not in config:
        return config
    result = dict(config)
    if ramp is not None:
        progress = board_dr_progress(config, iteration)
        for key in RANGE_KEYS:
            result[key] = [
                base + progress * (target - base)
                for base, target in zip(ramp["base_ranges"][key], config[key])
            ]
    if "length_ramp" in config:
        progress = board_length_progress(config, iteration)
        result["length_range_m"] = [
            base + progress * (target - base)
            for base, target in zip(
                config["length_ramp"]["start_range_m"], config["length_range_m"]
            )
        ]
    return result
