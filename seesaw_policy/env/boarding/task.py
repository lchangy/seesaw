"""Ground-to-board episode state; independent of simulator and policy layout."""

import torch


class BoardingTask:
    def __init__(self, count, device, config, dt):
        self.config, self.dt = config, dt
        for name in (
            "mount_hold",
            "balance_hold",
            "no_progress_time",
            "offboard_time",
            "previous_distance",
            "best_distance",
            "progress",
            "support_fraction",
            "elapsed",
            "mounted_time",
            "response_time",
            "response_good",
        ):
            setattr(self, name, torch.zeros(count, device=device))
        for name in (
            "mounted",
            "mount_event",
            "balance_achieved",
            "supported",
            "failure",
        ):
            setattr(self, name, torch.zeros(count, device=device, dtype=torch.bool))
        self.phase = torch.zeros(count, device=device, dtype=torch.long)
        self.window = torch.zeros(
            count, round(config["hold_seconds"] / dt), device=device, dtype=torch.bool
        )
        self.window_index = 0
        self.stable_fraction = torch.zeros(count, device=device)
        self.dynamic_success = torch.zeros(count, device=device, dtype=torch.bool)
        self.boarding_timeout = torch.zeros_like(self.dynamic_success)
        self.angle_progress = torch.zeros(count, device=device)
        self.previous_abs_angle = torch.zeros_like(self.angle_progress)
        self.angle_progress_valid = torch.zeros_like(self.dynamic_success)

    def reset(self, ids, entry_distance):
        for name in (
            "mount_hold",
            "balance_hold",
            "no_progress_time",
            "offboard_time",
            "progress",
            "support_fraction",
            "mounted",
            "mount_event",
            "balance_achieved",
            "supported",
            "failure",
            "phase",
            "elapsed",
            "mounted_time",
            "response_time",
            "response_good",
            "dynamic_success",
            "boarding_timeout",
        ):
            getattr(self, name)[ids] = 0
        self.previous_distance[ids] = entry_distance
        self.best_distance[ids] = entry_distance
        self.window[ids] = False
        self.stable_fraction[ids] = 0.0
        self.angle_progress[ids] = 0.0
        self.previous_abs_angle[ids] = 0.0
        self.angle_progress_valid[ids] = False

    def update(
        self,
        entry_distance,
        center_distance,
        board_contacts,
        ground_contacts,
        inside_board,
        below_board,
        outside_board,
        angle,
        rate,
        tilt,
        base_failure,
        enabled=None,
    ):
        if enabled is None:
            enabled = torch.ones_like(self.mounted)
        c, dt = self.config, self.dt * enabled
        self.elapsed += dt
        # Accept alternating diagonal support; mixed ground support is boarding,
        # not a completed mount. Brief contact gaps decay evidence gradually.
        mount_valid = (
            enabled
            & inside_board
            & (board_contacts >= 2)
            & (ground_contacts == 0)
            & (tilt < 1.0472)
        )
        evidence = torch.where(
            mount_valid, self.mount_hold + dt, (self.mount_hold - dt).clamp_min(0.0)
        )
        self.mount_hold.copy_(
            torch.where(enabled & inside_board & (ground_contacts == 0), evidence, 0.0)
        )
        self.mount_event.copy_(
            ~self.mounted & (self.mount_hold >= c["mount_hold_seconds"]) & ~base_failure
        )
        old_mounted = self.mounted.clone()
        self.mounted |= self.mount_event
        self.supported.copy_(enabled & inside_board & (board_contacts >= 2))
        contacts = board_contacts + ground_contacts
        self.support_fraction.copy_(
            torch.where(
                ~enabled,
                0.0,
                torch.where(
                    contacts > 0,
                    board_contacts.float() / contacts.clamp_min(1),
                    self.support_fraction,
                ),
            )
        )

        # Signed differences telescope within each phase: backtracking repays
        # progress. The switch has zero reward and a fresh distance baseline.
        distance = torch.where(old_mounted, center_distance, entry_distance)
        self.progress.copy_(self.previous_distance - distance)
        self.progress *= enabled
        distance = torch.where(self.mounted, center_distance, entry_distance)
        self.progress[self.mount_event] = 0.0
        self.previous_distance.copy_(distance)
        self.best_distance[self.mount_event] = distance[self.mount_event]

        balancing = (
            self.mounted & self.supported & (center_distance < c["balance_radius"])
        )
        improved = distance < self.best_distance - c["progress_epsilon"]
        self.best_distance.copy_(torch.where(improved, distance, self.best_distance))
        self.no_progress_time.copy_(
            torch.where(
                improved | self.mounted | ~enabled, 0.0, self.no_progress_time + dt
            )
        )
        # Mixed support is legal while boarding. Once mounted, returning to
        # ground/under the board must persist briefly; airborne != failure.
        dropped = self.mounted & ((ground_contacts >= 2) | below_board)
        self.offboard_time.copy_(torch.where(dropped, self.offboard_time + dt, 0.0))
        self.boarding_timeout.copy_(
            enabled & (self.elapsed >= c["boarding_deadline"] - 1e-5) & ~self.mounted
        )
        self.failure.copy_(
            enabled
            & (
                base_failure
                | self.boarding_timeout
                | (self.mounted & outside_board)
                | (self.offboard_time >= c["drop_hold_seconds"])
                | (self.no_progress_time >= c["no_progress_seconds"])
            )
        )
        # Signed angular progress (radians), not angular velocity. Establish a
        # fresh baseline on support acquisition/reset; never credit free motion.
        valid_angle = enabled & self.supported & ~self.failure & torch.isfinite(angle)
        magnitude = torch.nan_to_num(angle.abs(), nan=0.0, posinf=0.0, neginf=0.0)
        self.angle_progress.copy_(
            torch.where(
                valid_angle & self.angle_progress_valid,
                self.previous_abs_angle - magnitude,
                0.0,
            )
        )
        self.previous_abs_angle.copy_(magnitude)
        self.angle_progress_valid.copy_(valid_angle)
        # Use a full five-second occupancy window instead of resetting all
        # evidence on one noisy contact/angle frame.  Success still requires
        # the current frame to be stable, so an unstable terminal frame cannot
        # pass by relying on older samples in the window.
        stable = (
            self.mounted
            & self.supported
            & (angle.abs() < c["success_angle"])
            & ~self.failure
        )
        self.window[:, self.window_index] = stable
        self.window_index = (self.window_index + 1) % self.window.shape[1]
        self.stable_fraction.copy_(self.window.float().mean(dim=-1))
        self.balance_hold.copy_(self.window.sum(dim=-1) * self.dt)
        self.mounted_time.copy_(
            torch.where(self.mounted & ~self.failure, self.mounted_time + dt, 0.0)
        )
        enough_history = self.mounted_time >= c["hold_seconds"] - 1e-5
        self.balance_achieved |= (
            enough_history
            & stable
            & (self.stable_fraction >= c["success_window_fraction"])
        )
        self.dynamic_success.copy_(self.balance_achieved & stable)
        self.phase.copy_(
            torch.where(
                balancing, 2, torch.where(self.mounted | (board_contacts > 0), 1, 0)
            )
        )

    def balance_reward(self, center_distance, angle):
        return (self.mounted & self.supported & ~self.failure).float() * torch.exp(
            -center_distance.square() / self.config["balance_radius"] ** 2
            - angle.square() / 0.12**2
        )

    def record_response(self, command_x, velocity_x, command_age):
        # Ignore the first 150ms after each command switch. Zero command cannot
        # satisfy this metric; unsupported command time remains in the
        # denominator so losing contact cannot improve the response score.
        eligible = (
            self.mounted
            & ~self.mount_event
            & ~self.failure
            & (command_x.abs() > 0.15)
            & (command_age >= 0.15)
        )
        self.response_time += eligible * self.dt
        self.response_good += (
            eligible & self.supported & (velocity_x * command_x.sign() > 0.05)
        ) * self.dt
