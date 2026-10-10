"""Board balance shaping, settling and dimensionless edge risk."""

import math
import torch

REWARD_CONFIG = dict(
    reward_version=2,
    balance_angle_sigma=0.25,
    stable_angle_sigma=0.08726646,
    stable_rate_sigma=0.35,
    edge_safe_fraction=0.8,
    foot_radius=0.022,
    reward_scales=dict(
        seesaw_level=2.0,
        seesaw_center=1.0,
        seesaw_stable=2.0,
        seesaw_edge=-3.0,
        seesaw_four_foot=0.2,
    ),
)


SETTLING_REWARD_CONFIG = dict(
    REWARD_CONFIG,
    reward_version=3,
    stable_angle_sigma=math.radians(2.0),
    stable_rate_sigma=0.12,
    rate_braking_angle_sigma=math.radians(5.0),
    rate_near_level_multiplier=5.0,
    hold_rate_limit=0.12,
    hold_reward_seconds=2.0,
    reward_scales=dict(REWARD_CONFIG["reward_scales"], seesaw_hold=1.0),
)


RELATIVE_TRACKING_REWARD_CONFIG = dict(
    SETTLING_REWARD_CONFIG,
    reward_version=4,
    reward_scales=dict(
        SETTLING_REWARD_CONFIG["reward_scales"], seesaw_level=4.0, seesaw_stable=0.0
    ),
    angle_tracking_relative_sigma=0.25,
    stable_angle_relative_sigma=1.0 / 15.0,
    stable_rate_relative_sigma=0.25,
    rate_braking_angle_relative_sigma=1.0 / 6.0,
    tracking_response_seconds=1.0,
)


def relative_tracking_errors(angle, rate, max_angle, config):
    # Both targets are zero. Normalize by the board's actual excursion and
    # the reference speed for traversing that excursion in one response time.
    return angle / max_angle, rate / (max_angle / config["tracking_response_seconds"])


def four_foot_contact_term(boarding, board_contacts):
    active = boarding.mounted & boarding.supported & ~boarding.failure
    return (active & (board_contacts >= 4)).float()


def settling_terms(boarding, angle, rate, hold_seconds, config, dt, *, max_angle=None):
    active = boarding.mounted & boarding.supported & ~boarding.failure
    steady = (
        active
        & (angle.abs() < config["success_angle"])
        & (rate.abs() < config["hold_rate_limit"])
    )
    # Bound the clock and restart evidence whenever balance/support is lost.
    hold_seconds.copy_(
        torch.where(
            steady, (hold_seconds + dt).clamp(max=config["hold_reward_seconds"]), 0.0
        )
    )
    if config.get("reward_version", 1) >= 4:
        angle_error, rate_error = relative_tracking_errors(
            angle, rate, max_angle, config
        )
        braking_sigma = config["rate_braking_angle_relative_sigma"]
        angle_sigma = config["stable_angle_relative_sigma"]
        rate_sigma = config["stable_rate_relative_sigma"]
    else:
        angle_error, rate_error = angle, rate
        braking_sigma = config["rate_braking_angle_sigma"]
        angle_sigma, rate_sigma = (
            config["stable_angle_sigma"],
            config["stable_rate_sigma"],
        )
    near = torch.exp(-(angle_error / braking_sigma).square())
    braking = 1 + (config["rate_near_level_multiplier"] - 1) * near
    stable = torch.exp(
        -(angle_error / angle_sigma).square() - (rate_error / rate_sigma).square()
    )
    return dict(
        seesaw_rate=rate.square() * braking * active,
        seesaw_hold=stable * (hold_seconds / config["hold_reward_seconds"]) * active,
    )


def balance_terms(boarding, local, angle, rate, config, *, max_angle=None):
    active = boarding.mounted & boarding.supported & ~boarding.failure
    if config.get("reward_version", 1) >= 4:
        angle, rate = relative_tracking_errors(angle, rate, max_angle, config)
        level_sigma = config["angle_tracking_relative_sigma"]
        angle_sigma, rate_sigma = (
            config["stable_angle_relative_sigma"],
            config["stable_rate_relative_sigma"],
        )
    else:
        level_sigma = config["balance_angle_sigma"]
        angle_sigma, rate_sigma = (
            config["stable_angle_sigma"],
            config["stable_rate_sigma"],
        )
    level = torch.exp(-(angle / level_sigma).square())
    near_level = torch.exp(-(angle / angle_sigma).square())
    center = torch.exp(-local[:, :2].square().sum(-1) / config["balance_radius"] ** 2)
    stable = near_level * torch.exp(-(rate / rate_sigma).square())
    return dict(
        seesaw_level=level * active,
        seesaw_center=center * near_level * active,
        seesaw_stable=stable * active,
    )


def edge_risk(local, feet_local, half_width, config, *, half_length=None):
    # Project foot spheres onto the board; use the maximum foot risk so one
    # foot near an edge is not diluted by the other three. Airborne feet retain
    # a landing-margin signal. This is gated on mounting by the caller.
    half_length = (
        torch.full_like(half_width, config["half_length"])
        if half_length is None
        else half_length
    )
    extent = torch.stack((half_length, half_width), -1)
    safe = config["edge_safe_fraction"]
    base_excess = torch.relu(local[:, :2].abs() / extent - safe) / (1.0 - safe)
    foot_excess = torch.relu(
        (feet_local[:, :, :2].abs() + config["foot_radius"]) / extent[:, None] - safe
    ) / (1.0 - safe)
    return base_excess.square().sum(-1) + foot_excess.square().sum(-1).amax(-1)
