"""Seesaw task shaping and robot regularization terms."""

TASK_SCALES = dict(
    seesaw_progress=1.0,
    seesaw_mount=2.0,
    seesaw_level=2.0,
    seesaw_rate=-0.1,
    seesaw_edge=-3.0,
    seesaw_failure=-5.0,
    seesaw_tracking=1.0,
    seesaw_speed_excess=-1.0,
    seesaw_angle_progress=2.0,
    seesaw_success=5.0,
    seesaw_four_foot=0.2,
)
AUX_NAMES = {
    "torques",
    "dof_vel",
    "dof_acc",
    "action_rate",
    "collision",
    "dof_pos_limits",
    "torque_limits",
    "contact_forces",
    "base_acc",
    "roll_orientation",
    "correct_base_height",
}
