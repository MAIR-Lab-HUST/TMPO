# TreeMatch-RL

> **Tree-based Distribution Matching Online RL for Diverse and Efficient Diffusion Model Alignment**

## 核心特性

- **Softmax-TB 分布匹配** — 替代传统奖励最大化, 通过 GFlowNet 轨迹平衡实现多模态覆盖
- **三阶 27 分支树状采样** — 前缀共享降低计算开销, 仅 3 个 SDE 步注入噪声
- **Beta 分布自适应调度** — 根据在线奖励均值动态调整分叉位置
- **DPM-Solver++ Flash** — ODE 尾段时间步压缩, 加速 ~1.3x
- **RatioNorm IS** — 消除 Flow Matching 中 IS 偏置, 支持多次策略更新
- **低 GPU 友好** — LoRA + FSDP + Accelerate, 1×24GB 即可运行

## 项目结构

```
treematch-rl/
├── accelerate_configs/         # Accelerate 分布式配置
│   ├── single_gpu.yaml         # 单卡
│   ├── ddp_offload.yaml        # 2卡 DDP
│   └── fsdp_small.yaml         # 4卡 FSDP
├── config/                     # 训练配置
│   ├── sd35m_lora.yaml         # SD3.5-Medium
│   └── flux_lora.yaml          # Flux
├── treematch/                  # 核心代码
│   ├── train.py                # 主训练入口
│   ├── sampling/               # 采样模块
│   │   ├── sde_step.py         # SDE 噪声步进 + log_prob
│   │   ├── dpm_solver.py       # DPM-Solver++ 二阶
│   │   ├── scheduler.py        # Beta 分布自适应调度
│   │   └── tree_sampler.py     # 三阶 27 分支树状采样器
│   ├── losses/                 # 损失函数
│   │   ├── softmax_tb.py       # Softmax-TB 核心损失
│   │   ├── ratio_norm.py       # RatioNorm IS
│   │   ├── entropy.py          # RBF 粒子熵正则
│   │   ├── reference.py        # 参考模型约束
│   │   └── total_loss.py       # 总损失组装
│   ├── rewards/                # 奖励模型
│   │   ├── compute.py          # 多奖励并行计算 + 融合
│   │   ├── hpsv2.py            # HPSv2
│   │   ├── clipscore.py        # CLIP Score
│   │   └── aesthetic.py        # Aesthetic Score
│   └── utils/                  # 工具函数
│       ├── distributed.py      # 分布式通信
│       ├── checkpoint.py       # 检查点管理
│       └── logging_.py         # 日志工具
└── scripts/                    # 启动脚本
    ├── train_sd35m.sh
    ├── train_flux.sh
    └── eval.sh
```

## 快速开始

### 环境安装

```bash
pip install torch accelerate diffusers transformers peft open_clip_torch pyyaml
```

### 训练 SD3.5-Medium

```bash
# 单卡
accelerate launch --config_file accelerate_configs/single_gpu.yaml \
    treematch/train.py --config config/sd35m_lora.yaml

# 4卡 FSDP
accelerate launch --config_file accelerate_configs/fsdp_small.yaml \
    --num_processes 4 \
    treematch/train.py --config config/sd35m_lora.yaml
```

### 训练 Flux

```bash
accelerate launch --config_file accelerate_configs/fsdp_small.yaml \
    --num_processes 4 \
    treematch/train.py --config config/flux_lora.yaml
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `loss.beta` | 15.0 | Softmax-TB 温度 (越大越贪心) |
| `tree.kappa` | 4.0 | Beta 分布集中度 (0=均匀基线) |
| `tree.base_noise_levels` | [0.4, 0.7, 1.0] | 递增 SDE 噪声 |
| `loss.is_num_updates` | 4 | 每次采样后的 IS 更新次数 |
| `dpm_flash.compress_ratio` | 0.4 | ODE 尾段压缩比 |
| `training.learning_rate` | 1e-5 | 学习率 |
| `model.lora.rank` | 32 | LoRA rank |

## GPU 需求

| 配置 | 最低 GPU | 推荐 |
|------|---------|------|
| SD3.5-M + LoRA | 1×24GB | 4×24GB (FSDP) |
| Flux + LoRA | 2×24GB | 4×80GB (FSDP) |
| SD3.5-M 全参数 | 4×24GB | 8×80GB |
