from types import MethodType


def install_viewer_ray_cache(env):
    """Reuse packed terrain until the next sensor refresh; never cache physics."""
    if env.stage != 3:
        return
    runtime = env.runtimes[0]
    reader = runtime.rays
    original_read = reader.read
    original_sense = runtime.env.sim.sense
    cache = {}

    def sense(*args, **kwargs):
        cache.clear()
        env._viewer_packed_terrain = None
        return original_sense(*args, **kwargs)

    def read(*, purpose="state", pack=True):
        key = (purpose, pack)
        if key not in cache:
            if not pack and (purpose, True) in cache:
                cache[key] = (cache[(purpose, True)][0], None)
            else:
                cache[key] = original_read(purpose=purpose, pack=pack)
        if pack:
            env._viewer_packed_terrain = cache[key][1]
        return cache[key]

    def enable_terrain(_env):
        _env.terrain_data.data = reader.read()[0]
        return _env.terrain_data

    runtime.env.sim.sense = sense
    reader.read = read
    env.enable_terrain_data = MethodType(enable_terrain, env)
