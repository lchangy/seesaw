"""Foot-support roughness measured relative to a fitted local ground plane."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class FootholdPlaneResidual:
    """Reuse footmap rays inside a disk; missing samples provide no edge evidence.

    The fit removes height offsets and planar slopes, matching the Gym foothold
    reward's geometry. Its design and pseudoinverse are cached once per pool.
    """

    def __init__(self, axis: Sequence[float], radius: float, device: torch.device):
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Foothold radius must be finite and positive")
        coordinates = torch.as_tensor(axis, dtype=torch.float64)
        if coordinates.ndim != 1 or not torch.isfinite(coordinates).all():
            raise ValueError("Footmap axis must be a finite one-dimensional sequence")
        self.grid_shape = (len(coordinates), len(coordinates))
        x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")
        xy = torch.stack((x.flatten(), y.flatten()), dim=-1)
        selected = (xy.square().sum(-1) <= radius**2 + 1e-9).nonzero().flatten()
        local_xy = xy[selected] / radius
        design = torch.cat((local_xy, torch.ones_like(local_xy[:, :1])), dim=-1)
        if len(selected) < 3 or int(torch.linalg.matrix_rank(design)) != 3:
            raise ValueError(
                "Foothold disk must contain at least three noncollinear rays"
            )
        self.indices = selected.to(device=device)
        self.design = design.to(device=device, dtype=torch.float32)
        self.pinv = torch.linalg.pinv(design).to(device=device, dtype=torch.float32)

    def __call__(self, height: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if height.shape != valid.shape or tuple(height.shape[-2:]) != self.grid_shape:
            raise ValueError(
                "Footmap height/valid shapes must match the configured grid"
            )
        samples = height.flatten(-2).index_select(-1, self.indices)
        available = valid.flatten(-2).index_select(-1, self.indices).bool()
        available = available & torch.isfinite(samples)
        # Sanitize before fitting so NaNs from missed rays cannot contaminate
        # the batch. Like Gym, a partially missing disk gets no edge penalty.
        samples = torch.where(available, samples, 0.0)
        fitted = (samples @ self.pinv.T) @ self.design.T
        residual = (samples - fitted).square().mean(-1).sqrt()
        return torch.where(available.all(-1), residual, 0.0)
