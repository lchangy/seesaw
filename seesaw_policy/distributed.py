"""Shared parameter and pre-clipping gradient synchronization for distillation."""

import torch
from torch import distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def synchronize_gradients(parameters, sample_count=None):
    if not dist.is_initialized():
        return
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return
    flat = _flatten_dense_tensors(gradients)
    if sample_count is not None:
        flat.mul_(sample_count)
    dist.all_reduce(flat)
    if sample_count is None:
        flat.div_(dist.get_world_size())
    else:
        count = torch.tensor(float(sample_count), device=flat.device)
        dist.all_reduce(count)
        flat.div_(count)
    for gradient, averaged in zip(gradients, _unflatten_dense_tensors(flat, gradients)):
        gradient.copy_(averaged)


@torch.no_grad()
def synchronize_parameters(module):
    if dist.is_initialized():
        for value in module.state_dict().values():
            dist.broadcast(value, src=0)
