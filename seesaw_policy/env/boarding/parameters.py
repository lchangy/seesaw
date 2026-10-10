"""Single source of truth for immutable physical assets and command curriculum."""

import math
import torch

WIDTHS = (0.80, 0.70, 0.60, 0.50, 0.45, 0.40, 0.38, 0.35, 0.33, 0.30)
# Alternate geometry and command changes rather than increasing everything at once.
ANGLES_DEG = tuple(10.0 + 20.0 * i / 9 for i in range(10))
# The low underside touches z=0 at the joint stop, not a suspended 10cm step.
HEIGHTS = tuple(
    1.8 * math.sin(math.radians(a)) + 0.02 * math.cos(math.radians(a))
    for a in ANGLES_DEG
)
DAMPING = (1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 0.2, 0.2, 0.05, 0.05)
APPROACH_X = (0.6, 0.6, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5)
APPROACH_Y = (0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3)
BOARD_X = (0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0)
MOVE_PROB = (0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8)

# Command sampling is deliberately independent of the physical board level.
# A short perturbation is held for 0.5s and the next perturbation is scheduled
# 2s later; the level curriculum changes geometry only.
APPROACH_VX_MEAN = 0.8
APPROACH_VX_STD = 0.35
APPROACH_VX_MAX = max(APPROACH_X)
APPROACH_VY_STD = 0.12
APPROACH_VY_MAX = max(APPROACH_Y)
BOARD_VX_MAG_MEAN = 0.8
BOARD_VX_MAG_STD = 0.16
BOARD_VX_MAX = max(BOARD_X)
BOARD_COMMAND_INTERVAL_SECONDS = 2.0


def reset_angle(height):
    return math.asin(height / math.hypot(1.8, 0.02)) - math.atan2(0.02, 1.8)


def lookup(values, levels):
    return torch.as_tensor(values, device=levels.device, dtype=torch.float)[levels]


def _truncated_normal(count, mean, std, low, high, device):
    """Bounded Gaussian without level-dependent speed curriculum."""
    value = mean + std * torch.randn(count, device=device)
    for _ in range(8):
        invalid = (value < low) | (value > high)
        replacement = mean + std * torch.randn(count, device=device)
        value = torch.where(invalid, replacement, value)
    return value.clamp(low, high)


def sample_commands(levels, mounted):
    # ``levels`` remains in the signature for checkpoint/API compatibility;
    # speed ranges are intentionally not indexed by it.
    commands = torch.zeros(len(levels), 3, device=levels.device)
    count = len(levels)
    approach_vx = _truncated_normal(
        count, APPROACH_VX_MEAN, APPROACH_VX_STD, 0.0, APPROACH_VX_MAX, levels.device
    )
    approach_vy = _truncated_normal(
        count, 0.0, APPROACH_VY_STD, -APPROACH_VY_MAX, APPROACH_VY_MAX, levels.device
    )
    # Approach tracking remains active, but mounting switches to a zero
    # velocity command.  Periodic on-board pulses fought the uninterrupted
    # five-second balance objective and repeatedly reset its hold timer.
    commands[:, 0] = torch.where(mounted, 0.0, approach_vx)
    commands[:, 1] = torch.where(mounted, 0.0, approach_vy)
    return commands
