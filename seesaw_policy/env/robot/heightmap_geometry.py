"""Effective ground-ray limits, including historical checkpoint recipes."""


def heightmap_search_limits(geometry):
    # Old recipes start only 35 cm above the body, inside tall hurdle boxes.
    # Cast above all current support tops; retain range to deep gap-down pits.
    search = geometry["search"]
    ceiling = max(float(search["height_ceiling"]), 1.5)
    distance = max(float(search["height_range"]), ceiling + 3.0)
    return ceiling, distance


def footmap_search_limits(geometry):
    search = geometry["search"]
    ceiling = max(float(search["foot_ceiling"]), 1.5)
    distance = max(float(search["foot_range"]), ceiling + 3.0)
    return ceiling, distance
