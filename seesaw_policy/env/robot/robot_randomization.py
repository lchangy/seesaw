"""Iteration-based robot domain randomization ramp."""


def progress(domain_rand, iteration):
    config = domain_rand.get("robot_dr_ramp")
    if config is None:
        return 0.0
    start = config.get("start_iteration")
    duration = config.get("ramp_iterations")
    return min(max((iteration - start) / duration, 0.0), 1.0)
