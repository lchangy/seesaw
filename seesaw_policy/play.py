"""Play one seesaw policy environment with adjustable board geometry."""

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from .camera import MJlabDepthCamera
from .config import SensorConfig
from .model import SeesawPolicy
from .train import SCHEMA


class Controller:
    def __init__(self, model, env, sensor, compile_mode="none"):
        self.model, self.env, self.sensor = model, env, sensor
        self.camera = MJlabDepthCamera(env, sensor)
        self.state = model.rssm.initial(env.num_envs)
        self.previous = torch.zeros(env.num_envs, 12, device=env.device)
        self.first = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.image = torch.full(
            (env.num_envs, sensor.camera_height, sensor.camera_width, 1),
            0.5,
            device=env.device,
        )
        self.features = model.encode_image(self.image)
        self.tick = 0
        self.step = (
            model.step
            if compile_mode == "none"
            else torch.compile(model.step, mode=compile_mode)
        )

    @torch.no_grad()
    def action(self, obs):
        self.camera.reset_envs(self.first)
        capture = self.tick % self.sensor.camera_interval == 0
        if capture or self.first.any():
            fresh = self.camera.capture()
            self.image = (
                fresh
                if capture
                else torch.where(self.first[:, None, None, None], fresh, self.image)
            )
            self.features = self.model.encode_image(self.image)
        action, self.state = self.step(
            obs, self.features, self.previous, self.first, self.state
        )
        return action.clamp(
            -self.env.cfg.normalization.clip_actions,
            self.env.cfg.normalization.clip_actions,
        )

    def advance(self, action, done):
        self.previous = action.detach()
        self.first = done.bool()
        self.tick += 1

    def reset(self):
        self.first[:] = True
        self.previous.zero_()
        self.image.fill_(0.5)
        self.features.zero_()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead"),
        default="default",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("Steps must be positive")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("schema") != SCHEMA:
        parser.error("Select a seesaw policy PPO checkpoint")
    from seesaw_policy.env.robot.config import TaskConfig
    from .env.vector_env import SeesawEnv
    from seesaw_policy.env.viewer_support import _ViewerEnv

    env = SeesawEnv(
        1,
        args.device,
        recipe=TaskConfig(**saved["recipe"]),
        seed=saved["options"]["seed"],
        playback=True,
    )
    try:
        state = saved["rank_states"][0]["curriculum"]
        env.seesaw_runtime.board_curriculum.load_state_dict(state["mjlab_seesaw"])
        env.set_training_iteration(saved["iteration"])
        env.reset()
        sensor = replace(
            SensorConfig(**saved["sensor"]),
            depth_augmentation=False,
            camera_mount_dr=False,
        )
        model = SeesawPolicy(
            sensor, saved["memory"], saved["privileged_width"], args.device
        )
        model.load_state_dict(saved["model"])
        model.eval()
        controller = Controller(model, env, sensor, args.compile_mode)
        if args.viewer:
            import viser
            from seesaw_policy.env.board.viewer import SeesawViewer
            from seesaw_policy.env.sensor_cache import install_viewer_ray_cache

            install_viewer_ray_cache(env)
            runtime = env.seesaw_runtime
            runtime.play_levels = runtime.levels.clone()

            class ViewerEnv(_ViewerEnv):
                def step(self, action):
                    result = env.step(action)
                    controller.advance(action, result[3])
                    return result

                def reset(self):
                    controller.reset()
                    return env.reset()

            @torch.no_grad()
            def policy(obs):
                return controller.action(obs)

            policy.reset = controller.reset
            policy.runner = SimpleNamespace(
                history=torch.zeros(1, 26, 45, device=args.device)
            )
            server = viser.ViserServer(
                host=args.host, port=args.port, label="seesaw policy"
            )
            viewer = SeesawViewer(ViewerEnv(env), policy, viser_server=server)
            print(f"SEESAW_POLICY_PLAY url=http://{args.host}:{args.port}", flush=True)
            viewer.run()
        else:
            reward = 0.0
            with torch.inference_mode():
                for _ in range(args.steps):
                    action = controller.action(env.get_observations())
                    _, _, r, done, _ = env.step(action)
                    controller.advance(action, done)
                    reward += float(r.mean())
            print(
                f"SEESAW_POLICY_PLAY steps={args.steps} mean_reward={reward/args.steps:.6f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
