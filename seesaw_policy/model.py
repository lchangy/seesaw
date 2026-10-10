"""Randomly initialized depth encoder, recurrent state and policy; no label model."""
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.dreamer.networks import MultiEncoder, RSSM
from .vendor.dreamer.tools import OneHotDist

class CategoricalState(OneHotDist):
    """Known softmax logits: avoid per-step Python/GPU constraint checks.

    Training still checks finite losses/gradients before an optimizer update.
    This is local to Seesaw policy, not a global distributions setting.
    """
    def __init__(self, logits, unimix_ratio):
        if unimix_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            logits = (probs*(1-unimix_ratio)+unimix_ratio/probs.shape[-1]).log()
        torch.distributions.OneHotCategorical.__init__(self, logits=logits, validate_args=False)

    def mode(self):
        # The vendor's super().logits property is not traceable by Dynamo.
        mode = F.one_hot(self.logits.argmax(-1), self.logits.shape[-1])
        # Deterministic views need the same bounded probability surrogate as
        # sampled one-hot states. A logit straight-through path amplifies
        # gradients at every recurrent step and can explode over a rollout.
        return mode.detach()+(self.probs-self.probs.detach())

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError('Seesaw policy uses the per-rank torch RNG, not a sample seed')
        sample = torch.distributions.OneHotCategorical.sample(self, sample_shape)
        probs = self.probs
        while probs.ndim < sample.ndim:
            probs = probs[None]
        return sample+(probs-probs.detach())

class RecurrentState(RSSM):
    """Same parameters and recurrence, without GPU-dependent Python branches."""
    def get_dist(self, state, dtype=None):
        if not self._discrete:
            return super().get_dist(state, dtype)
        return torch.distributions.Independent(
            CategoricalState(state['logit'], self._unimix_ratio), 1, validate_args=False)

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True, *, reset_state=None):
        initial = reset_state if reset_state is not None else self.initial(len(is_first))
        if prev_state is None:
            prev_state = initial
            prev_action = torch.zeros_like(prev_action)
        else:
            first = is_first.bool()
            prev_action = torch.where(first[:, None], 0., prev_action)
            prev_state = {k: torch.where(first.reshape(-1, *([1]*(v.ndim-1))), initial[k], v)
                          for k, v in prev_state.items()}
        prior = self.img_step(prev_state, prev_action)
        stats = self._suff_stats_layer('obs', self._obs_out_layers(torch.cat((prior['deter'], embed), -1)))
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        return {'stoch': stoch, 'deter': prior['deter'], **stats}, prior


def mlp(widths):
    layers = []
    for index, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if index < len(widths)-2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class SeesawPolicy(nn.Module):
    def __init__(self, sensor, memory, privileged_width, device='cpu'):
        super().__init__()
        self.encoder = MultiEncoder({'prop': (33,), 'image':
            (sensor.camera_height, sensor.camera_width, 1)}, use_camera=True, **memory['encoder'])
        cnn = self.encoder._cnn
        channels = [m.out_channels for m in cnn.layers if isinstance(m, nn.Conv2d)][-1]
        original_width = cnn.outdim
        cnn.layers.add_module('spatial_pool', nn.AdaptiveAvgPool2d((2, 2)))
        cnn.outdim = channels*4
        self.encoder.outdim += cnn.outdim-original_width
        self.rssm = RecurrentState(stoch=memory['dyn_stoch'], deter=memory['dyn_deter'],
            hidden=memory['dyn_hidden'], rec_depth=memory['dyn_rec_depth'],
            discrete=memory['dyn_discrete'], act=memory['act'], norm=memory['norm'],
            unimix_ratio=memory['unimix_ratio'], initial=memory['initial'],
            num_actions=12, embed=self.encoder.outdim, device=str(device))
        width = memory['dyn_deter']+memory['dyn_stoch']*memory['dyn_discrete']
        self.projector = mlp([width+45, 512, 256, 64])
        self.actor = mlp([109, 256, 128, 64, 12])
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.zeros_(self.actor[-1].bias)
        self.critic = mlp([45+privileged_width, 512, 256, 128, 2])
        self.log_std = nn.Parameter(torch.zeros(12))
        self.to(device)

    def encode_image(self, image):
        return self.encoder._cnn(image+.5)

    def step(self, obs, image_features, previous, first, state):
        embedding = torch.cat((image_features, self.encoder._mlp(obs[:, :33])), -1)
        state, _ = self.rssm.obs_step(state, previous, embedding, first, sample=False)
        features = self.rssm.get_feat(state)
        latent = F.normalize(self.projector(torch.cat((features, obs), -1)), dim=-1)
        return self.actor(torch.cat((latent, obs), -1)), state

    def value(self, obs, privileged):
        return self.critic(torch.cat((obs, privileged), -1))

    def sequence(self, obs, image, previous, first, state):
        # Replay complete environment sequences; resets mask BPTT at episode boundaries.
        batch, length = obs.shape[:2]
        encoded = self.encode_image(image.flatten(0, 1)).reshape(batch, length, -1)
        means = []
        for time in range(length):
            mean, state = self.step(obs[:, time], encoded[:, time], previous[:, time],
                                    first[:, time], state)
            means.append(mean)
        return torch.stack(means, 1)
