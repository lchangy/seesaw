"""Load the standalone seesaw configuration, without other training stages."""

from seesaw_policy.env.robot.config import load_task_config


def make_task_recipe():
    return load_task_config()
