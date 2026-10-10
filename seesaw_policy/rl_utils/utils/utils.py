# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from typing import Tuple

import numpy as np
import torch


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self._torch_mean = None
        self._torch_var = None
        self._torch_count = None

    def _clear_torch_stats(self) -> None:
        self._torch_mean = None
        self._torch_var = None
        self._torch_count = None

    def _sync_numpy_from_torch(self) -> None:
        if self._torch_mean is None:
            return
        self.mean = self._torch_mean.detach().cpu().numpy().astype(np.float64, copy=True)
        self.var = self._torch_var.detach().cpu().numpy().astype(np.float64, copy=True)
        self.count = float(self._torch_count.detach().cpu().item())

    def _torch_stats(self, device):
        device = torch.device(device)
        if self._torch_mean is None or self._torch_mean.device != device:
            self._torch_mean = torch.as_tensor(self.mean, device=device, dtype=torch.float32)
            self._torch_var = torch.as_tensor(self.var, device=device, dtype=torch.float32)
            self._torch_count = torch.as_tensor(self.count, device=device, dtype=torch.float32)
        return self._torch_mean, self._torch_var, self._torch_count

    def update(self, arr: np.ndarray) -> None:
        self._sync_numpy_from_torch()
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        self._sync_numpy_from_torch()
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count
        self._clear_torch_stats()

    @torch.no_grad()
    def update_torch(self, arr: torch.Tensor) -> None:
        batch = arr.detach().to(dtype=torch.float32)
        batch_mean = torch.mean(batch, dim=0)
        batch_var = torch.var(batch, dim=0, unbiased=False)
        batch_count = batch.shape[0]

        mean, var, count = self._torch_stats(batch.device)
        delta = batch_mean - mean
        total_count = count + float(batch_count)
        new_mean = mean + delta * float(batch_count) / total_count
        m_a = var * count
        m_b = batch_var * float(batch_count)
        m_2 = m_a + m_b + torch.square(delta) * count * float(batch_count) / total_count

        self._torch_mean = new_mean.detach()
        self._torch_var = (m_2 / total_count).detach()
        self._torch_count = total_count.detach()

    def state_dict(self):
        self._sync_numpy_from_torch()
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state):
        mean = state["mean"]
        var = state["var"]
        if isinstance(mean, torch.Tensor):
            mean = mean.detach().cpu().numpy()
        if isinstance(var, torch.Tensor):
            var = var.detach().cpu().numpy()
        self.mean = np.asarray(mean, dtype=np.float64)
        self.var = np.asarray(var, dtype=np.float64)
        self.count = float(state["count"])
        self._clear_torch_stats()


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0, group_slices=None):
        """Running per-coordinate normalization with optional group balancing.

        ``RunningMeanStd`` removes the physical units and offsets of each
        coordinate.  For observations made of heterogeneous groups, however,
        a 12-D group would still contribute roughly 12 times the squared
        energy of a 1-D group after standardization.  ``group_slices`` uses
        ``1/sqrt(group_width)`` scaling so every group has comparable expected
        L2 energy while preserving the relative information inside a group.

        Each entry is ``(name, start, end)`` in the normalized observation
        (end-exclusive).  Unassigned coordinates retain scale one, which
        keeps this class backwards compatible with the regular PPO normalizer.
        """
        input_dim = int(input_dim)
        super().__init__(shape=(input_dim,))
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.group_slices = tuple(group_slices or ())
        self.group_names = tuple(str(group[0]) for group in self.group_slices)
        self._group_scale = None
        self._group_scale_torch_cache = {}
        if self.group_slices:
            group_scale = np.ones(input_dim, dtype=np.float32)
            occupied = np.zeros(input_dim, dtype=np.bool_)
            for group in self.group_slices:
                if len(group) != 3:
                    raise ValueError(
                        "Normalizer group entries must be (name, start, end)"
                    )
                _, start, end = group
                start, end = int(start), int(end)
                if start < 0 or end > input_dim or end <= start:
                    raise ValueError(
                        f"Invalid normalizer group slice ({start}, {end}) "
                        f"for input_dim={input_dim}"
                    )
                if occupied[start:end].any():
                    raise ValueError(
                        f"Overlapping normalizer group slice ({start}, {end})"
                    )
                occupied[start:end] = True
                group_scale[start:end] = 1.0 / np.sqrt(float(end - start))
            self._group_scale = group_scale

    def _scale_numpy(self, normalized):
        if self._group_scale is not None:
            normalized = normalized * self._group_scale
        return normalized

    def _scale_torch(self, normalized):
        if self._group_scale is None:
            return normalized
        key = (normalized.device, normalized.dtype)
        scale = self._group_scale_torch_cache.get(key)
        if scale is None:
            scale = torch.as_tensor(
                self._group_scale,
                device=normalized.device,
                dtype=normalized.dtype,
            )
            self._group_scale_torch_cache[key] = scale
        return normalized * scale

    def normalize(self, input):
        self._sync_numpy_from_torch()
        normalized = (input - self.mean) / np.sqrt(self.var + self.epsilon)
        normalized = self._scale_numpy(normalized)
        return np.clip(normalized, -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch, var_torch, _ = self._torch_stats(device)
        std_torch = torch.sqrt(var_torch + self.epsilon)
        normalized = (input - mean_torch) / std_torch
        normalized = self._scale_torch(normalized)
        return torch.clamp(normalized, -self.clip_obs, self.clip_obs)


def quaternion_slerp(q0, q1, fraction, shortestpath=True):
    if fraction.dim() == 1:
        fraction = fraction.unsqueeze(-1)
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    if shortestpath:
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = torch.abs(dot)
    dot = torch.clamp(dot, -1.0, 1.0)

    linear = dot > 0.9995
    lerp = torch.nn.functional.normalize((1.0 - fraction) * q0 + fraction * q1, dim=-1)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp(min=1e-8)
    theta = theta_0 * fraction
    s0 = torch.cos(theta) - dot * torch.sin(theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    slerp = torch.nn.functional.normalize(s0 * q0 + s1 * q1, dim=-1)
    return torch.where(linear, lerp, slerp)

def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)
