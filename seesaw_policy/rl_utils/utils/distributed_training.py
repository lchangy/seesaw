"""Explicit collectives for replicated learners; inert without an initialized group.

Differentiable statistics below assume every rank evaluates the SAME function of
the global statistic and subsequently averages parameter gradients with
``sync_gradients``. Their local derivatives are scaled by world size to cancel
that average. They are not general-purpose autograd communication operators.
Launch, device assignment, initial/resumed state broadcast and RNGs belong to
the caller. No process group is created here.
"""
import torch


def _distributed():
    # Some IsaacGym Torch builds omit c10d (and even the distributed module).
    try:
        import torch.distributed as dist
    except (ImportError, AttributeError):
        return None
    if not getattr(dist, "is_available", lambda: False)():
        return None
    if not getattr(dist, "is_initialized", lambda: False)():
        return None
    return dist


def active():
    dist = _distributed()
    return dist is not None and dist.get_world_size() > 1


def rank():
    dist = _distributed()
    return 0 if dist is None else dist.get_rank()


def world_size():
    dist = _distributed()
    return 1 if dist is None else dist.get_world_size()


def local_quota(global_count, offset=0):
    """Shard a GLOBAL budget; offset rotates remainders for grouped budgets."""
    count = int(global_count)
    if count < 0:
        raise ValueError("Global sample budget must be nonnegative")
    base, remainder = divmod(count, world_size())
    return base + int((rank() - int(offset)) % world_size() < remainder)


def mean_from_local(local_mean, count):
    """Global mean from a local mean/count; empty means may be NaN.

    Select zero BEFORE multiplying by count: NaN * 0 would otherwise poison
    the global sum. The empty contribution has zero derivative as well.
    """
    count = torch.as_tensor(count, device=local_mean.device, dtype=local_mean.dtype).detach()
    local_mean = torch.where(count > 0, local_mean, torch.zeros_like(local_mean))
    if not active():
        return local_mean
    total = sum_detached(count)
    return replicated_sum(local_mean * count) / total.clamp_min(1)


def sum_detached(value):
    result = value.detach().clone()
    if active():
        _distributed().all_reduce(result)
    return result


def broadcast(value, src=0):
    if active():
        _distributed().broadcast(value, src=src)
    return value


def replicated_sum(value):
    """Global forward sum, local derivative scaled for averaged gradients."""
    if not active():
        return value
    return sum_detached(value) + world_size() * (value - value.detach())


def mean(value, dim=None):
    """Global element mean, or global row mean when dim=0; empty-safe."""
    if dim not in (None, 0):
        raise ValueError("Distributed mean supports dim=None or dim=0")
    if not active():
        if (value.numel() if dim is None else value.shape[0]) == 0:
            return value.sum(dim=dim)
        return value.mean(dim=dim)
    count = value.numel() if dim is None else value.shape[0]
    total_count = sum_detached(value.new_tensor(float(count)))
    return replicated_sum(value.sum(dim=dim)) / total_count.clamp_min(1)


def weighted_mean(value, weights, min_weight=1e-8):
    # Difficulty/eligibility weights are statistics, not learned parameters.
    weights = weights.detach()
    denominator = sum_detached(weights.sum()).clamp_min(min_weight)
    return replicated_sum((value * weights).sum()) / denominator


def gather_rows(value, differentiable=False):
    """Concatenate variable-size rank shards, including empty shards.

    differentiable=True is only for replicated global losses (see module doc).
    All trailing dimensions, dtype and device type must agree across ranks.
    """
    if not active():
        return value
    dist = _distributed()
    size = torch.tensor([len(value)], device=value.device, dtype=torch.long)
    sizes = [torch.empty_like(size) for _ in range(world_size())]
    dist.all_gather(sizes, size)
    sizes = [int(item.item()) for item in sizes]
    padded = value.detach().new_zeros((max(max(sizes), 1),) + value.shape[1:])
    padded[:len(value)].copy_(value.detach())
    shards = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(shards, padded)
    rows = [shard[:count] for shard, count in zip(shards, sizes)]
    if differentiable:
        rows[rank()] = value.detach() + world_size() * (value - value.detach())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def sync_gradients(optimizer):
    """Average all optimizer-owned gradients before clipping, including None.

    Fixed dtype/device buckets and a presence reduction keep collective order
    identical for rank-local unused parameters. Globally unused parameters stay
    None, preserving Adam's skip semantics. Shared parameters occur only once.
    """
    if not active():
        return
    buckets, seen = {}, set()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            buckets.setdefault((parameter.device, parameter.dtype), []).append(parameter)
    dist = _distributed()
    for parameters in buckets.values():
        present = torch.tensor([p.grad is not None for p in parameters],
                               device=parameters[0].device, dtype=torch.int32)
        dist.all_reduce(present)
        used = present.tolist()
        selected = [p for p, count in zip(parameters, used) if count]
        if not selected:
            continue
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in selected])
        dist.all_reduce(flat)
        flat.div_(world_size())
        offset = 0
        for parameter in selected:
            grad = flat[offset:offset + parameter.numel()].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = grad.clone()
            else:
                parameter.grad.copy_(grad)
            offset += parameter.numel()


def critic_balance_loss(critic):
    """Consume existing conditional-critic gates without modifying its module."""
    if not active():
        return critic.pop_balance_loss()
    gates = (torch.cat(critic.gate_cache) if critic.gate_cache else
             critic.value_head.weight.new_empty((0, critic.num_experts)))
    critic.gate_cache.clear()
    usage = mean(gates, dim=0)
    critic.last_usage = usage.detach()
    critic.last_entropy = mean(-(gates * gates.clamp_min(1e-8).log()).sum(1)).detach()
    count = sum_detached(gates.new_tensor(float(len(gates))))
    return (critic.num_experts * usage.square().sum() - 1) * (count > 0)


@torch.no_grad()
def update_normalizer(normalizer, batch, reference=False):
    """Merge new global moments once; never sum replicated historical state."""
    if not active():
        if reference:
            normalizer.update(batch.detach().cpu().numpy())
        else:
            normalizer.update_torch(batch)
        return
    batch = batch.detach().double()
    count = sum_detached(batch.new_tensor(float(len(batch))))
    if count.item() == 0:
        return
    batch_mean = sum_detached(batch.sum(0)) / count
    batch_var = sum_detached((batch - batch_mean).square().sum(0)) / count
    normalizer.update_from_moments(batch_mean.cpu().numpy(), batch_var.cpu().numpy(),
                                   int(count.item()))


@torch.no_grad()
def broadcast_normalizer(normalizer, device):
    if not active():
        return
    state = normalizer.state_dict()
    for key in ('mean', 'var', 'count'):
        value = torch.as_tensor(state[key], device=device, dtype=torch.float64).clone()
        _distributed().broadcast(value, src=0)
        state[key] = value.cpu().numpy() if key != 'count' else value.item()
    normalizer.load_state_dict(state)
