# Seesaw Policy

Go2 翘翘板视觉策略：深度图和本体观测 → 时序记忆 → 动作，通过 recurrent PPO 从零训练。

## 安装

使用 Python 3.12 和支持 CUDA 的环境，在仓库根目录执行：

```bash
python -m pip install -e .
```

## 训练

```bash
seesaw-policy-train --envs 8192 --iterations 10000 --compile-mode default
```

三卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m seesaw_policy.train --envs 8192 --iterations 10000 --compile-mode default
```

每轮采集轨迹、计算 reward 和 advantage，再更新视觉编码器、时序记忆、actor 和 critic。根据成功率推进翘翘板课程。默认开启编译，首次运行需要预热；日志和 checkpoint 保存在 `logs/seesaw_policy/`。

## 续训

保持与原训练相同的 GPU 数和环境数；`--iterations` 为额外训练轮数。三卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m seesaw_policy.train --resume /path/to/model_1000.pt \
  --envs 8192 --iterations 1000 --compile-mode default
```

## Play

```bash
seesaw-policy-play --checkpoint /path/to/model_1000.pt --viewer --compile-mode default
```

打开 http://127.0.0.1:8082 。单环境运行，启动时暂停；调节板长、板宽、倾角或 level 后点击 Reset，再点击 Play。
