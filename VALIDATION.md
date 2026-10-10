# Local validation

2026-10-10, `student` branch, Python 3.12 and CUDA in `/home/orion/.venvs/go2-mjlab`.

- Editable install (`pip install --no-deps --no-build-isolation -e .`) and both installed CLI `--help` checks passed.
- `python -m unittest tests.test_seesaw_policy tests.test_seesaw_rewards`: 8 tests passed. Covers recurrent replay consistency, encoder/memory/actor/critic gradients, terminal/timeout returns, seesaw-only allocation and relative balance rewards.
- Native training: 8 seesaw environments, 2 control steps, one epoch/minibatch, one PPO iteration with `--compile-mode default`; saved `model_1.pt`. First iteration took about 32.5 s including compilation, not a throughput benchmark.
- Resume: loaded model, optimizer and curriculum, completed iteration 2 with `--compile-mode none` and saved `model_2.pt`.
- Compiled headless playback: one environment, three steps, finite output/rewards; passed.
- Native viewer setup, one seesaw world, geometry override (2.8 m length, 0.4 m width, 10° tilt), reset and one policy-controlled physics step passed. The temporary viewer was closed.

Artifacts and logs are outside the repository under `/tmp/seesaw_student_*`. These are random-initialization smoke policies, not trained balance policies. No pretrained policy is bundled. Full 8192-environment, three-GPU training and long-run convergence have not been tested or launched.

Editable installation with the repository robot/reset assets is supported. Resume requires the same GPU count and total environments; episodes and rollout memory restart. `main` preserves the initial independent extraction, while `student` contains the cleaned runtime namespace. This branch does not support the original teacher/distillation entrypoints or their checkpoints.
