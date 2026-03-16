# TreeMatch-RL

> **Tree-based Distribution Matching Online RL for Diverse and Efficient Diffusion Model Alignment**

## 核心特性

- **Softmax-TB 分布匹配** — 替代传统奖励最大化, 通过 GFlowNet 轨迹平衡实现多模态覆盖
- **三阶 27 分支树状采样** — 前缀共享降低计算开销, 仅 3 个 SDE 步注入噪声
- **Beta 分布自适应调度** — 根据在线奖励均值动态调整分叉位置与噪声系数
- **DPM-Solver++ Flash** — ODE 尾段时间步压缩, 加速 ~1.3x
- **RatioNorm IS** — 消除 Flow Matching 中 E[log w] < 0 偏置, 支持多次策略更新
- **8×H200 优化** — FSDP SHARD_GRAD_OP (ZeRO-2) + LoRA + bf16

---

## 项目结构

```
treematch-rl/
├── accelerate_configs/         # Accelerate 分布式配置
│   ├── single_gpu.yaml         # 单卡
│   ├── ddp_offload.yaml        # 2 卡 DDP
│   ├── fsdp_small.yaml         # 4 卡 FSDP FULL_SHARD
│   └── fsdp_8gpu.yaml          # 8×H200 SHARD_GRAD_OP (ZeRO-2)  ← 推荐
├── config/                     # 训练配置 (YAML)
│   ├── sd35m_lora.yaml         # SD3.5-Medium LoRA
│   └── flux_lora.yaml          # Flux LoRA
├── treematch/                  # 核心代码
│   ├── train.py                # 主训练入口 + 训练循环
│   ├── sampling/
│   │   ├── sde_step.py         # SDE 噪声步进 + log_prob 计算
│   │   ├── dpm_solver.py       # DPM-Solver++ 二阶求解器
│   │   ├── scheduler.py        # Beta 分布自适应调度器
│   │   └── tree_sampler.py     # 三阶 27 分支树状采样器
│   ├── losses/
│   │   ├── softmax_tb.py       # Softmax-TB 核心损失
│   │   ├── ratio_norm.py       # RatioNorm IS 权重计算
│   │   ├── entropy.py          # RBF 粒子熵正则化
│   │   ├── reference.py        # 参考模型约束损失
│   │   └── total_loss.py       # 总损失组装 + 训练诊断指标
│   ├── rewards/
│   │   ├── compute.py          # 多奖励并行计算 + advantage_aggr 融合
│   │   ├── hpsv2.py            # HPSv2 (人类偏好评分)
│   │   ├── clipscore.py        # CLIP Score (文图一致性)
│   │   └── aesthetic.py        # Aesthetic Score (美学评分)
│   └── utils/
│       ├── distributed.py      # AllGather + per-prompt 归一化
│       ├── checkpoint.py       # LoRA / 全参数检查点管理
│       └── logging_.py         # 日志工具
├── scripts/
│   ├── train_8xH200.sh         # 8×H200 启动脚本  ← 推荐
│   ├── train_sd35m.sh
│   ├── train_flux.sh
│   └── eval.sh
└── data/
    └── pickapic_prompts.json   # Prompt 数据 (需自备)
```

---

## 环境搭建 (完整步骤)

```bash
# 1. 创建 conda 环境
conda create -n treematch python=3.10 -y
conda activate treematch

# 2. 安装 PyTorch (CUDA 12.1)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装核心依赖
pip install accelerate==0.33.0 diffusers==0.30.0 transformers==4.44.0

# 4. 安装 LoRA + 奖励模型依赖
pip install peft open_clip_torch

# 5. 安装工具库
pip install pyyaml Pillow numpy

# 6. 验证 GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.0f} GB)')
"
```

---

## 数据集

训练使用 JSON 格式的 prompt 列表。每步从中采样一条 prompt, 生成 27 条路径并计算奖励。

**支持格式**:

```json
// 格式 A: 纯列表
["a photo of a cat", "a painting of mountains", ...]

// 格式 B: 字典列表
[{"prompt": "a photo of a cat"}, {"prompt": "a painting of mountains"}, ...]

// 格式 C: 键值对
{"001": "a photo of a cat", "002": "a painting of mountains", ...}
```

**推荐数据源**: [Pick-a-Pic](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2) 的 prompt 子集, 包含多样化文本描述。

配置路径:
```yaml
dataset:
  data_json_path: "data/pickapic_prompts.json"
  resolution: [512, 512]   # H×W, SD3.5 推荐 512; Flux 推荐 720
```

---

## 奖励模型设计

TreeMatch-RL 采用**多奖励模型并行计算 + advantage_aggr 融合**:

### 支持的奖励模型

| 模型 | 评估维度 | 数值范围 | 权重 |
|------|---------|---------|------|
| **HPSv2** | 人类偏好 (构图、细节、整体质量) | ~[0.2, 0.35] | 0.6 |
| **CLIP Score** | 文图一致性 (prompt 匹配度) | ~[20, 35] | 0.4 |
| **Aesthetic** | 美学评分 (可选) | ~[4, 8] | - |

### 融合策略: `advantage_aggr` (推荐)

各奖励模型**独立归一化后加权合并**, 解决数值范围差异:

```
HPSv2:     [0.25, 0.28, 0.30, ...]  → 归一化 → [-1.2, 0.1, 1.1, ...]  × 0.6
CLIP:      [22.1, 25.3, 28.7, ...]  → 归一化 → [-1.0, 0.2, 1.5, ...]  × 0.4
                                                                         ─────
最终优势:                                                        加权求和 → (K,)
```

### 奖励权重准备

```bash
mkdir -p reward_ckpt

# HPSv2 权重 (必须)
# 下载 HPS_v2.1_compressed.pt 和 open_clip_pytorch_model.bin 到 reward_ckpt/

# CLIP Score (自动从 HuggingFace 下载)
# 配置: clip_score_path: "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
```

---

## 8×H200 训练 (推荐)

### 快速启动

```bash
# 默认配置 (SD3.5-Medium LoRA)
bash scripts/train_8xH200.sh

# 指定配置 + CLI 覆盖
bash scripts/train_8xH200.sh config/sd35m_lora.yaml --lr 2e-5 --max_steps 500

# 或手动指定
accelerate launch \
    --config_file accelerate_configs/fsdp_8gpu.yaml \
    --num_processes 8 \
    treematch/train.py \
    --config config/sd35m_lora.yaml \
    --lr 1e-5 --beta 15.0
```

### 分布式配置说明

`fsdp_8gpu.yaml` 使用 **SHARD_GRAD_OP** (ZeRO-2):

| 配置项 | 值 | 说明 |
|--------|---|------|
| `fsdp_sharding_strategy` | `SHARD_GRAD_OP` | 梯度+优化器状态分片, 模型参数全量保留 |
| `fsdp_backward_prefetch` | `BACKWARD_PRE` | 反向传播预取, 提升带宽利用率 |
| `fsdp_activation_checkpointing` | `true` | 激活值检查点, 节省显存 |
| `mixed_precision` | `bf16` | H200 原生支持 bf16 |
| `fsdp_use_orig_params` | `true` | LoRA 兼容必须 |

### Accelerate Launch 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config_file` | accelerate 配置文件 | `accelerate_configs/fsdp_8gpu.yaml` |
| `--num_processes` | GPU 数量 (覆盖 yaml) | `8` |
| `--mixed_precision` | 混合精度类型 | `bf16` / `fp16` |
| `--main_process_port` | 主进程端口 (多任务避免冲突) | `29500` |
| `--gpu_ids` | 指定 GPU 编号 | `0,1,2,3,4,5,6,7` |

---

## CLI 参数详解 (含 GRPO 对比)

所有 YAML 配置项均可通过命令行 `--key value` 覆盖, **CLI 优先级高于 YAML**。

### Training 参数

| CLI 参数 | YAML 路径 | 默认值 | 说明 | GRPO-Guard | MixGRPO | flow_grpo |
|----------|-----------|--------|------|------------|---------|-----------|
| `--lr` | `training.learning_rate` | `1e-5` | 学习率 | 1e-5 | 1e-4 | 3e-5 |
| `--max_steps` | `training.max_train_steps` | `300` | 最大训练步数 | 300 | 1000 | 200 |
| `--grad_accum` | `training.gradient_accumulation_steps` | `3` | 梯度累积步数 | 1 | 2 | 4 |
| `--batch_size` | `training.vae_decode_batch_size` | `4` | VAE 分批解码大小 | 4 | 8 | 4 |
| `--seed` | `training.seed` | `42` | 随机种子 | 42 | 42 | 42 |
| `--max_grad_norm` | `training.max_grad_norm` | `1.0` | 梯度裁剪范数 | 1.0 | 1.0 | 1.0 |
| `--output_dir` | `training.output_dir` | `outputs/sd35m` | 输出目录 | - | - | - |

### Loss 参数

| CLI 参数 | YAML 路径 | 默认值 | 说明 | GRPO-Guard | MixGRPO | flow_grpo |
|----------|-----------|--------|------|------------|---------|-----------|
| `--beta` | `loss.beta` | `15.0` | Softmax-TB 温度 β (越大越贪心) | 0.01 (KL β) | - (PPO clip) | 0.04 (KL β) |
| `--lambda_entropy` | `loss.lambda_entropy` | `0.01` | 粒子熵正则权重 λ₁ | - | - | - |
| `--lambda_ref` | `loss.lambda_ref` | `0.1` | 参考约束权重 λ₂ | 0.01 (KL) | - | 0.04 (KL) |
| `--is_clip_range` | `loss.is_clip_range` | `0.2` | IS 权重裁剪范围 ε | 0.2 (PPO clip) | 0.2 | 0.2 |
| `--is_num_updates` | `loss.is_num_updates` | `4` | 每次采样后 IS 更新次数 | 4 | 1 | 4 |

> **注**: GRPO-Guard 的 `beta` 是 KL 散度系数 (越小约束越弱), TreeMatch-RL 的 `beta` 是 Softmax-TB 温度 (越大越趋向奖励最大化, β=0 等于均匀分布)。两者含义完全不同。

### Tree 参数

| CLI 参数 | YAML 路径 | 默认值 | 说明 | TreeGRPO |
|----------|-----------|--------|------|----------|
| `--kappa` | `tree.kappa` | `4.0` | Beta 分布集中度 (0=均匀基线) | - |
| `--num_inference_steps` | `tree.num_inference_steps` | `28` | 总采样步数 | 28 |
| `--tree_k` | `tree.k` | `3` | 每步分支数 (3→27 路径) | 2 (→4 路径) |

---

## 训练指标说明

训练过程中每步打印两行日志:

```
[Step 1/300] loss=12.35 tb=11.23 entropy=0.08 ref=0.00 reward=0.25 α=0.50
             approx_kl=0.0023 clipfrac=0.12 ratio=1.002±0.045 log_prob=-1234.5 grad_norm=0.87
```

### 指标解读

| 指标 | 含义 | 健康范围 | 异常信号 & 处理 |
|------|------|---------|----------------|
| **loss** | 总损失 | 稳步下降 | 发散 → 降低 `--lr` |
| **tb** | Softmax-TB 加权损失 | 稳步下降 | 不下降 → 调整 `--beta` |
| **entropy** | RBF 粒子熵 (路径相似度) | 缓慢下降 | 不降 = 路径多样性好 |
| **ref** | 参考约束损失 | 保持低值 | 过大 → 策略偏离, 增大 `--lambda_ref` |
| **reward** | 平均奖励 | 稳步上升 | 不动 → 检查奖励模型权重路径 |
| **α** | 难度水平 (Beta 调度) | 0→1 缓慢上升 | 一直 0 = 奖励太低; 一直 1 = 太高 |
| **approx_kl** | 策略偏移程度 | < 0.1 | > 0.5 → 减少 `--is_num_updates` |
| **clipfrac** | IS 权重被裁剪比例 | < 0.3 | > 0.5 → 增大 `--is_clip_range` |
| **ratio** | IS ratio 均值±标准差 | 1.0 ± 0.1 | 远离 1.0 → RatioNorm 异常 |
| **log_prob** | 当前策略 log_prob 均值 | 缓慢变化 | 骤变 → 模型不稳定 |
| **grad_norm** | 梯度范数 (clip 后) | < `max_grad_norm` | 持续被 clip → 降低 `--lr` |

---

## GPU 需求

| 配置 | 最低 GPU | 推荐 |
|------|---------|------|
| SD3.5-M + LoRA | 1×24GB | 4×24GB (FSDP) |
| Flux + LoRA | 2×24GB | 4×80GB (FSDP) |
| SD3.5-M 全参数 | 4×24GB | 8×80GB |
| **大规模训练 (推荐)** | 4×80GB | **8×H200 80GB (SHARD_GRAD_OP)** |
