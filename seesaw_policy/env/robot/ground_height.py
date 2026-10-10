"""Ground-relative height shared by Stage2 rewards and AMP observations."""

import torch


def local_base_height(heightmap, x_indices, y_indices, *, center_surface=False):
    """Return vertical base clearance and validity; misses never use spawn Z.

    Heightmap values are hit_z - base_z, not ray travel distances. Mask before
    reducing so invalid NaNs cannot contaminate otherwise valid samples.
    """
    height = heightmap["height"].index_select(1, x_indices).index_select(2, y_indices)
    valid = heightmap["valid"].index_select(1, x_indices).index_select(2, y_indices)
    valid = valid & torch.isfinite(height)
    count = valid.sum((1, 2))
    if center_surface:
        # At a hurdle/bridge edge, averaging the top and the pit invents a
        # support plane. Use the ground directly below the base instead;
        # the nearest valid ray is the fallback if that central ray misses.
        nx, ny = height.shape[1:]
        x = torch.arange(nx, device=height.device) - nx // 2
        y = torch.arange(ny, device=height.device) - ny // 2
        distance = (x[:, None].square() + y[None, :].square()).expand_as(height)
        selected = distance.masked_fill(~valid, nx * nx + ny * ny).flatten(1).argmin(1)
        clearance = (
            -torch.where(valid, height, 0.0)
            .flatten(1)
            .gather(1, selected[:, None])
            .squeeze(1)
        )
        return clearance, count > 0
    clearance = -torch.where(valid, height, 0.0).sum((1, 2)) / count.clamp_min(1)
    return clearance, count > 0
