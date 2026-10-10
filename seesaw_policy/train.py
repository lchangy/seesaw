"""Train seesaw policy from random initialization using recurrent PPO."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from .camera import MJlabDepthCamera
from .config import SensorConfig, memory_defaults
from .distributed import synchronize_gradients, synchronize_parameters
from .model import SeesawPolicy

SCHEMA = "seesaw_policy_ppo_v1"


def returns(rewards, values, dones, timeouts, bootstrap, gamma=0.99, lam=0.95):
    # Environment auto-resets: preserve the shared runtime's pre-step value
    # timeout convention; true falls receive no bootstrap.
    rewards = rewards + gamma * values * (dones & timeouts)[..., None]
    advantage = torch.zeros_like(bootstrap)
    targets = torch.empty_like(values)
    next_value = bootstrap
    for t in reversed(range(len(rewards))):
        active = (~dones[t]).float()[:, None]
        delta = rewards[t] + gamma * next_value * active - values[t]
        advantage = delta + gamma * lam * active * advantage
        targets[t] = advantage + values[t]
        next_value = values[t]
    return targets


class Runner:
    def __init__(self, env, args, sensor, memory):
        self.env, self.args = env, args
        self.sensor, self.memory = sensor, memory
        self.model = SeesawPolicy(sensor, memory, env.num_privileged_obs, env.device)
        synchronize_parameters(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate
        )
        self.camera = MJlabDepthCamera(env, sensor)
        self.state = self.model.rssm.initial(env.num_envs)
        self.previous = torch.zeros(env.num_envs, 12, device=env.device)
        self.first = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.image = torch.full(
            (env.num_envs, sensor.camera_height, sensor.camera_width, 1),
            0.5,
            device=env.device,
        )
        self.tick = 0
        self.iteration = 0
        self.policy_step = self.model.step
        self.policy_sequence = self.model.sequence
        if args.compile_mode != "none":
            # Same functions for rollout and recurrent replay; no frozen feature cache.
            self.policy_step = torch.compile(self.model.step, mode=args.compile_mode)
            self.policy_sequence = torch.compile(
                self.model.sequence, mode=args.compile_mode
            )

    @torch.no_grad()
    def rollout(self):
        env, model = self.env, self.model
        initial = {k: v.detach().clone() for k, v in self.state.items()}
        rows = {
            k: []
            for k in (
                "obs",
                "privileged",
                "image",
                "previous",
                "first",
                "action",
                "log_prob",
                "mean",
                "value",
                "reward",
                "done",
                "timeout",
            )
        }
        image_feature = model.encode_image(self.image)
        for _ in range(self.args.rollout_steps):
            obs = env.get_observations().clone()
            privileged = env.get_privileged_observations().clone()
            self.camera.reset_envs(self.first)
            # Hold optical frames at 10 Hz; capture reset environments immediately.
            capture = self.tick % self.sensor.camera_interval == 0
            if capture or self.first.any():
                fresh = self.camera.capture()
                self.image = (
                    fresh
                    if capture
                    else torch.where(self.first[:, None, None, None], fresh, self.image)
                )
                image_feature = model.encode_image(self.image)
            mean, self.state = self.policy_step(
                obs, image_feature, self.previous, self.first, self.state
            )
            value = model.value(obs, privileged)
            distribution = Normal(mean, model.log_std.exp())
            action = distribution.sample()
            clip = float(env.cfg.normalization.clip_actions)
            executed = action.clamp(-clip, clip)
            _, _, _, done, infos = env.step(executed)
            done = done.bool()
            reward = torch.stack((infos["task_rewards"], infos["aux_rewards"]), -1)
            invalid = done & env.last_velocity_nonfinite_reset_buf.bool()
            reward = torch.where(invalid[:, None], 0.0, reward)
            items = (
                obs,
                privileged,
                self.image,
                self.previous,
                self.first,
                action,
                distribution.log_prob(action).sum(-1),
                mean,
                value,
                reward,
                done,
                infos["time_outs"].bool(),
            )
            for key, item in zip(rows, items):
                rows[key].append(item.detach().clone())
            self.previous = executed
            self.first = done
            self.tick += 1
        bootstrap = model.value(
            env.get_observations(), env.get_privileged_observations()
        )
        rows = {k: torch.stack(v) for k, v in rows.items()}
        rows["returns"] = returns(
            rows["reward"],
            rows["value"],
            rows["done"],
            rows["timeout"],
            bootstrap,
            self.args.gamma,
            self.args.gae_lambda,
        )
        algorithm = env.task_config.train_config["algorithm"]
        advantage = rows["returns"] - rows["value"]
        weights = advantage.new_tensor(
            [algorithm["task_advantage_weight"], algorithm["aux_advantage_weight"]]
        )
        # Normalize per reward channel over all ranks, weighted by actual shard size.
        stats = torch.stack(
            (
                advantage.sum((0, 1)),
                advantage.square().sum((0, 1)),
                torch.full(
                    (2,), advantage.shape[0] * advantage.shape[1], device=env.device
                ),
            )
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        mean = stats[0] / stats[2]
        std = (stats[1] / stats[2] - mean.square()).clamp_min(1e-8).sqrt()
        rows["advantage"] = (((advantage - mean) / std) * weights).sum(-1)
        # Preserve whole sequences; only the environment axis is shuffled for PPO.
        return {k: v.transpose(0, 1) for k, v in rows.items()}, initial

    def update(self, rows, initial):
        model, args = self.model, self.args
        totals = torch.zeros(4, device=self.env.device)
        count = 0
        for _ in range(args.epochs):
            for ids in torch.randperm(
                self.env.num_envs, device=self.env.device
            ).tensor_split(args.minibatches):
                state = {k: v[ids] for k, v in initial.items()}
                mean = self.policy_sequence(
                    rows["obs"][ids],
                    rows["image"][ids],
                    rows["previous"][ids],
                    rows["first"][ids],
                    state,
                )
                distribution = Normal(mean, model.log_std.exp())
                log_prob = distribution.log_prob(rows["action"][ids]).sum(-1)
                ratio = (log_prob - rows["log_prob"][ids]).exp()
                advantage = rows["advantage"][ids]
                policy_loss = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1 - args.clip, 1 + args.clip) * advantage,
                ).mean()
                values = model.value(rows["obs"][ids], rows["privileged"][ids])
                old = rows["value"][ids]
                clipped = old + (values - old).clamp(-args.clip, args.clip)
                value_loss = torch.maximum(
                    (values - rows["returns"][ids]).square(),
                    (clipped - rows["returns"][ids]).square(),
                ).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy_coef * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                loss.backward()
                synchronize_gradients(list(model.parameters()), sample_count=len(ids))
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                with torch.no_grad():
                    model.log_std.clamp_(math.log(0.03), math.log(1.2))
                totals += torch.stack(
                    (policy_loss, value_loss, entropy, model.log_std.exp().mean())
                ).detach()
                count += 1
        return dict(
            zip(
                ("loss/policy", "loss/value", "policy/entropy", "policy/std"),
                (totals / count).tolist(),
            )
        )

    def save(self, path):
        local = dict(
            curriculum=self.env.curriculum_state_dict(),
            rng=dict(
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(),
                python=random.getstate(),
            ),
        )
        ranks = [None] * self.env.world_size
        if torch.distributed.is_initialized():
            torch.distributed.all_gather_object(ranks, local)
        else:
            ranks[0] = local
        if self.env.rank == 0:
            saved = dict(
                schema=SCHEMA,
                iteration=self.iteration,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                recipe=asdict(self.env.task_config),
                sensor=asdict(self.sensor),
                memory=self.memory,
                options=vars(self.args),
                privileged_width=self.env.num_privileged_obs,
                rank_states=ranks,
                world_size=self.env.world_size,
                global_num_envs=self.env.global_num_envs,
            )
            temporary = path.with_suffix(".tmp")
            torch.save(saved, temporary)
            temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs", type=int, default=8192, help="Total environments across all GPUs"
    )
    parser.add_argument(
        "--iterations", type=int, default=10000, help="Additional PPO iterations"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead"),
        default="default",
    )
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--camera-pitch", type=float, default=5.0)
    args = parser.parse_args(argv)
    if (
        min(
            args.iterations,
            args.save_interval,
            args.rollout_steps,
            args.epochs,
            args.minibatches,
        )
        < 1
    ):
        parser.error(
            "Iterations, rollout, epochs, batches and save interval must be positive"
        )
    if not (
        0 < args.learning_rate
        and math.isfinite(args.learning_rate)
        and 0 < args.gamma <= 1
        and 0 <= args.gae_lambda <= 1
        and 0 < args.clip < 1
        and args.max_grad_norm > 0
        and args.entropy_coef >= 0
        and args.value_coef > 0
        and math.isfinite(args.camera_pitch)
    ):
        parser.error("Invalid PPO settings")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if args.envs // world < max(8, args.minibatches):
        parser.error("Each GPU needs at least eight environments and one per minibatch")
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        torch.distributed.init_process_group("nccl")
    from .task import make_task_recipe
    from seesaw_policy.env.robot.config import TaskConfig
    from .env.vector_env import SeesawEnv

    source = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    if source and source.get("schema") != SCHEMA:
        parser.error("Resume requires a seesaw policy PPO checkpoint")
    if source and (
        source["world_size"] != world or source["global_num_envs"] != args.envs
    ):
        parser.error("Resume preserves GPU count and environment count")
    if source:
        # PPO settings and camera geometry belong to the saved training run.
        for name in (
            "seed",
            "rollout_steps",
            "epochs",
            "minibatches",
            "learning_rate",
            "entropy_coef",
            "value_coef",
            "clip",
            "gamma",
            "gae_lambda",
            "max_grad_norm",
            "camera_pitch",
        ):
            setattr(args, name, source["options"][name])
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    recipe = TaskConfig(**source["recipe"]) if source else make_task_recipe()
    sensor = (
        SensorConfig(**source["sensor"])
        if source
        else SensorConfig(camera_pitch=args.camera_pitch, depth_augmentation=True)
    )
    memory = source["memory"] if source else memory_defaults()
    path = args.run_dir or Path("logs/seesaw_policy") / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    if world > 1:
        paths = [str(path) if rank == 0 else None]
        torch.distributed.broadcast_object_list(paths, src=0)
        path = Path(paths[0])
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
        (path / "run_manifest.json").write_text(
            json.dumps(
                dict(
                    task="seesaw policy",
                    training="from_scratch_recurrent_ppo",
                    checkpoint_source=str(args.resume) if source else None,
                    recipe=asdict(recipe),
                    sensor=asdict(sensor),
                    memory=memory,
                    options={
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                ),
                indent=2,
            )
            + "\n"
        )
    env = writer = None
    try:
        env = SeesawEnv(
            args.envs // world + int(rank < args.envs % world),
            args.device,
            recipe=recipe,
            seed=args.seed,
            rank=rank,
            world_size=world,
            global_num_envs=args.envs,
        )
        runner = Runner(env, args, sensor, memory)
        if source:
            runner.model.load_state_dict(source["model"])
            runner.optimizer.load_state_dict(source["optimizer"])
            runner.iteration = source["iteration"]
            env.load_curriculum_state_dict(source["rank_states"][rank]["curriculum"])
            rng = source["rank_states"][rank]["rng"]
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        env.set_training_iteration(runner.iteration)
        env.reset()
        if rank == 0:
            writer = SummaryWriter(str(path))
        end = runner.iteration + args.iterations
        while runner.iteration < end:
            start = time.perf_counter()
            env.set_training_iteration(runner.iteration)
            rows, initial = runner.rollout()
            metrics = runner.update(rows, initial)
            runner.iteration += 1
            metrics.update(
                {
                    "reward/task": float(rows["reward"][..., 0].mean()),
                    "reward/aux": float(rows["reward"][..., 1].mean()),
                    "performance/iteration_seconds": time.perf_counter() - start,
                }
            )
            board = env.seesaw_runtime.board_curriculum
            runtime = env.seesaw_runtime
            episode_counts = torch.tensor(
                [runtime.completed_episodes, runtime.completed_successes],
                device=env.device,
                dtype=torch.float64,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(episode_counts)
            episodes, successes = episode_counts.tolist()
            metrics.update(
                {
                    "seesaw/level": board.level,
                    "seesaw/window_episodes": board.last_episode_count,
                    "seesaw/window_success_rate": board.last_success_rate,
                    "seesaw/window_mount_rate": board.last_mount_rate,
                    "seesaw/completed_episodes": episodes,
                    "seesaw/completed_successes": successes,
                    "seesaw/success_rate": successes / max(episodes, 1),
                }
            )
            if writer:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, runner.iteration)
                writer.flush()
                print(
                    f"SEESAW_POLICY iteration={runner.iteration} "
                    + json.dumps(metrics),
                    flush=True,
                )
            if runner.iteration % args.save_interval == 0 or runner.iteration == end:
                runner.save(path / f"model_{runner.iteration}.pt")
    finally:
        if writer:
            writer.close()
        if env:
            env.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
