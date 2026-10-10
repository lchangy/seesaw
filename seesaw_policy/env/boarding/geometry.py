"""Batched analytic box queries. Each row queries only its own board."""

import torch


def rotate_y(v, angle):
    while angle.ndim < v.ndim - 1:
        angle = angle.unsqueeze(-1)
    c, s = angle.cos(), angle.sin()
    return torch.stack(
        (c * v[..., 0] + s * v[..., 2], v[..., 1], -s * v[..., 0] + c * v[..., 2]), -1
    )


def ray_box(origins, directions, center, angle, half, limits):
    o = rotate_y(origins - center[:, None], -angle)
    d = rotate_y(directions, -angle)
    half = torch.as_tensor(half, device=o.device, dtype=o.dtype)
    if half.ndim == 2:
        half = half[:, None, :]
    parallel = d.abs() < 1e-8
    safe = torch.where(parallel, torch.ones_like(d), d)
    a, b = (-half - o) / safe, (half - o) / safe
    lo = torch.where(parallel, torch.full_like(a, -float("inf")), torch.minimum(a, b))
    hi = torch.where(parallel, torch.full_like(a, float("inf")), torch.maximum(a, b))
    enter, axis_in = lo.max(-1)
    leave, axis_out = hi.min(-1)
    outside = (parallel & (o.abs() > half)).any(-1)
    inside = enter < 0
    distance = torch.where(inside, leave, enter)
    valid = ~outside & (leave >= enter) & (distance >= 0) & (distance <= limits)
    axis = torch.where(inside, axis_out, axis_in)
    normal = torch.zeros_like(o)
    # Orient against the ray, including rays starting inside the box.
    normal.scatter_(-1, axis[..., None], -d.gather(-1, axis[..., None]).sign())
    return (
        torch.where(valid, distance, torch.as_tensor(limits, device=o.device)),
        rotate_y(normal, angle) * valid[..., None],
        valid,
    )


def nearest_box(points, center, angle, half):
    p = rotate_y(points - center[:, None], -angle)
    half = torch.as_tensor(half, device=p.device, dtype=p.dtype)
    if half.ndim == 2:
        half = half[:, None, :]
    closest = p.clamp(-half, half)
    inside = (p.abs() <= half).all(-1)
    axis = (half - p.abs()).argmin(-1)
    sign = torch.where(p >= 0, torch.ones_like(p), -torch.ones_like(p))
    face = p.clone()
    face.scatter_(-1, axis[..., None], (half * sign).gather(-1, axis[..., None]))
    closest = torch.where(inside[..., None], face, closest)
    return rotate_y(closest - p, angle)
