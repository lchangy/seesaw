import glob
import json

import numpy as np
import torch

from seesaw_policy.rl_utils.utils import utils


class AMPLoader:
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    TAR_TOE_VEL_LOCAL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE
    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE
    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE
    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE
    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE
    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE
    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE
    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        motion_files=None,
        preload_transitions=False,
        num_preload_transitions=200000,
        num_frames=2,
        include_base_lin_vel=True,
    ):
        self.device = device
        self.num_frames = int(num_frames)
        if self.num_frames not in (2, 3):
            raise ValueError("AMP num_frames must be 2 or 3")
        # The base linear velocity is useful for speed-conditioned AMP, but it
        # can make a single high-speed expert dominate a low-speed command
        # curriculum.  Keep the full 61-D file layout unchanged and make this
        # feature optional only in the discriminator observation.
        self.include_base_lin_vel = bool(include_base_lin_vel)
        self.time_between_frames = float(time_between_frames)
        motion_files = sorted(motion_files or glob.glob("datasets/go2_motion/*"))
        if not motion_files:
            raise ValueError("AMP motion file list is empty")

        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for motion_idx, motion_file in enumerate(motion_files):
            with open(motion_file, "r", encoding="utf-8") as f:
                motion_json = json.load(f)
            motion_data = np.asarray(motion_json["Frames"], dtype=np.float32)
            if motion_data.shape[1] < self.TAR_TOE_VEL_LOCAL_END_IDX:
                raise ValueError(f"AMP motion {motion_file} has invalid frame dim {motion_data.shape[1]}")

            root_rot = motion_data[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]
            norm = np.linalg.norm(root_rot, axis=1, keepdims=True).clip(min=1e-8)
            root_rot = root_rot / norm
            root_rot[root_rot[:, -1] < 0.0] *= -1.0
            motion_data[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX] = root_rot

            frame_duration = float(motion_json["FrameDuration"])
            traj_len = (motion_data.shape[0] - 1) * frame_duration
            if traj_len < (self.num_frames - 1) * self.time_between_frames:
                raise ValueError(f"AMP motion too short for {self.num_frames} frames: {motion_file}")
            self.trajectories_full.append(torch.tensor(motion_data, dtype=torch.float32, device=device))
            self.trajectory_names.append(motion_file)
            self.trajectory_idxs.append(motion_idx)
            self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))
            self.trajectory_frame_durations.append(frame_duration)
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(int(motion_data.shape[0]))
            print(f"Loaded {traj_len:.3f}s AMP motion from {motion_file}.")

        self.trajectory_weights = np.asarray(self.trajectory_weights, dtype=np.float64)
        self.trajectory_weights /= self.trajectory_weights.sum()
        self.trajectory_frame_durations = np.asarray(self.trajectory_frame_durations, dtype=np.float64)
        self.trajectory_lens = np.asarray(self.trajectory_lens, dtype=np.float64)
        self.trajectory_num_frames = np.asarray(self.trajectory_num_frames, dtype=np.int64)

        # A zero preload request means "sample expert transitions on demand".
        # Treating it as an enabled empty cache made ``feed_forward_generator``
        # call ``torch.randint(0, 0, ...)`` inside AmpMoECTS and abort the
        # whole run before the first update.
        num_preload_transitions = int(num_preload_transitions)
        self.preload_transitions = bool(preload_transitions and num_preload_transitions > 0)
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} AMP transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_frames = tuple(
                self.get_full_frame_at_time_batch(traj_idxs, times + i * self.time_between_frames)
                for i in range(self.num_frames)
            )
            self.preloaded_s, self.preloaded_s_next = self.preloaded_frames[:2]
            print("Finished AMP transition preload")

    @property
    def observation_dim(self):
        return (
            self.JOINT_POS_SIZE
            + self.TAR_TOE_POS_LOCAL_SIZE
            + (self.LINEAR_VEL_SIZE if self.include_base_lin_vel else 0)
            + self.ANGULAR_VEL_SIZE
            + self.JOINT_VEL_SIZE
            + 1
        )

    @property
    def observation_groups(self):
        """Semantic groups in the 43-D or 40-D AMP observation.

        The order mirrors ``Go2Robot.get_amp_observations`` and
        ``_amp_obs_from_full_frame``: joint pose, local toe position, optional
        base linear velocity, base angular velocity, joint velocity, and base
        height.
        """
        offset = 0
        group_defs = [
            ("joint_pos", self.JOINT_POS_SIZE),
            ("foot_pos", self.TAR_TOE_POS_LOCAL_SIZE),
        ]
        if self.include_base_lin_vel:
            group_defs.append(("base_lin_vel", self.LINEAR_VEL_SIZE))
        group_defs.extend([
            ("base_ang_vel", self.ANGULAR_VEL_SIZE),
            ("joint_vel", self.JOINT_VEL_SIZE),
            ("base_height", 1),
        ])
        groups = []
        for name, width in group_defs:
            groups.append((name, offset, offset + width))
            offset += width
        return tuple(groups)

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_full_frame_batch(self, size):
        traj_idxs = self.weighted_traj_idx_sample_batch(size)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def weighted_traj_idx_sample_batch(self, size):
        return np.random.choice(
            self.trajectory_idxs,
            size=int(size),
            p=self.trajectory_weights,
            replace=True,
        )

    def traj_time_sample_batch(self, traj_idxs):
        subst = (self.num_frames - 1) * self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        frame_idx = times / self.trajectory_frame_durations[traj_idxs]
        idx_low = np.floor(frame_idx).astype(np.int64)
        idx_high = idx_low + 1
        blend = frame_idx - idx_low

        batch_size = len(traj_idxs)
        all_frame_pos_starts = torch.zeros(batch_size, self.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(batch_size, self.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(batch_size, self.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(batch_size, self.ROT_SIZE, device=self.device)
        amp_dim = self.JOINT_VEL_END_IDX - self.JOINT_POSE_START_IDX
        all_frame_amp_starts = torch.zeros(batch_size, amp_dim, device=self.device)
        all_frame_amp_ends = torch.zeros(batch_size, amp_dim, device=self.device)

        for traj_idx in set(traj_idxs):
            traj_mask_np = traj_idxs == traj_idx
            traj_mask = torch.as_tensor(traj_mask_np, dtype=torch.bool, device=self.device)
            trajectory = self.trajectories_full[traj_idx]
            max_idx = self.trajectory_num_frames[traj_idx] - 1
            low = torch.as_tensor(np.clip(idx_low[traj_mask_np], 0, max_idx), dtype=torch.long, device=self.device)
            high = torch.as_tensor(np.clip(idx_high[traj_mask_np], 0, max_idx), dtype=torch.long, device=self.device)
            starts = trajectory[low]
            ends = trajectory[high]
            all_frame_pos_starts[traj_mask] = starts[:, self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]
            all_frame_pos_ends[traj_mask] = ends[:, self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]
            all_frame_rot_starts[traj_mask] = starts[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]
            all_frame_rot_ends[traj_mask] = ends[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]
            all_frame_amp_starts[traj_mask] = starts[:, self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask] = ends[:, self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX]

        blend = torch.tensor(blend, dtype=torch.float32, device=self.device).unsqueeze(-1)
        pos_blend = self._lerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self._lerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(int(num_mini_batch)):
            if self.preload_transitions:
                idxs = torch.randint(
                    0,
                    self.preloaded_s.shape[0],
                    (int(mini_batch_size),),
                    device=self.device,
                )
                frames = tuple(frame[idxs] for frame in self.preloaded_frames)
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                frames = tuple(
                    self.get_full_frame_at_time_batch(traj_idxs, times + i * self.time_between_frames)
                    for i in range(self.num_frames)
                )
            yield tuple(self._amp_obs_from_full_frame(frame) for frame in frames)

    def _amp_obs_from_full_frame(self, frames):
        # Keep full-frame storage compatible with all existing motion files,
        # while selecting exactly the fields used by the discriminator.
        fields = [
            frames[:, self.JOINT_POSE_START_IDX:self.JOINT_POSE_END_IDX],
            frames[:, self.TAR_TOE_POS_LOCAL_START_IDX:self.TAR_TOE_POS_LOCAL_END_IDX],
        ]
        if self.include_base_lin_vel:
            fields.append(frames[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX])
        fields.extend([
            frames[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX],
            frames[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX],
            frames[:, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3],
        ])
        return torch.cat(fields, dim=-1)

    @staticmethod
    def get_root_pos_batch(poses):
        return poses[:, AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_linear_vel_batch(poses):
        return poses[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

    @staticmethod
    def get_angular_vel_batch(poses):
        return poses[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def _lerp(val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1
