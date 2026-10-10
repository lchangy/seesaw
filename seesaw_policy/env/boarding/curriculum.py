"""Globally aggregated seesaw curriculum; replay cannot promote levels."""

import torch


class SeesawCurriculum:
    def __init__(self, device, config):
        self.config = dict(config)
        self.level = 0
        self.counts = torch.zeros(10, 5, device=device)
        self.pending = torch.zeros_like(self.counts)
        self.bad_windows = 0
        self.last_episode_count = 0.0
        self.last_success_rate = 0.0
        self.last_mount_rate = 0.0
        self.last_delta = 0

    def record(
        self, levels, success, mounted=None, response_time=None, response_good=None
    ):
        mounted = success if mounted is None else mounted
        response_time = (
            torch.zeros_like(success) if response_time is None else response_time
        )
        response_good = (
            torch.zeros_like(success) if response_good is None else response_good
        )
        self.pending.index_add_(
            0,
            levels,
            torch.stack(
                (
                    torch.ones_like(success),
                    success,
                    mounted,
                    response_time,
                    response_good,
                ),
                -1,
            ).float(),
        )

    def update(self):
        from seesaw_policy.rl_utils.utils.distributed_training import sum_detached

        self.counts.add_(sum_detached(self.pending))
        self.pending.zero_()
        row = self.counts[self.level]
        if row[0] >= self.config["minimum_episodes"]:
            rate = float(row[1] / row[0])
            mount_rate = float(row[2] / row[0])
            self.last_episode_count = float(row[0])
            self.last_success_rate = rate
            self.last_mount_rate = mount_rate
            threshold = 0.7 if self.level < 4 else self.config["promote_rate"]
            promote = rate >= threshold and mount_rate >= (
                0.85 if self.level < 4 else 0.9
            )
            self.bad_windows = (
                self.bad_windows + 1 if rate < self.config["demote_rate"] else 0
            )
            delta = 1 if promote else -1 if self.bad_windows >= 2 else 0
            self.last_delta = delta
            # Every decision consumes one independent episode window.  Keeping
            # these counts made the next simulator step reuse the same failed
            # episodes and falsely turn one bad window into two.
            self.counts.zero_()
            self.level = min(9, max(0, self.level + delta))
            if delta:
                self.bad_windows = 0

    def state_dict(self):
        return dict(
            config=self.config,
            level=self.level,
            counts=self.counts.cpu(),
            bad_windows=self.bad_windows,
            last_episode_count=self.last_episode_count,
            last_success_rate=self.last_success_rate,
            last_mount_rate=self.last_mount_rate,
            last_delta=self.last_delta,
        )

    def load_state_dict(self, state):
        if state["config"] != self.config:
            raise ValueError("Seesaw task recipe differs from checkpoint")
        self.level = int(state["level"])
        self.counts.copy_(state["counts"])
        self.bad_windows = int(state.get("bad_windows", 0))
        self.last_episode_count = float(state.get("last_episode_count", 0.0))
        self.last_success_rate = float(state.get("last_success_rate", 0.0))
        self.last_mount_rate = float(state.get("last_mount_rate", 0.0))
        self.last_delta = int(state.get("last_delta", 0))
        self.pending.zero_()
