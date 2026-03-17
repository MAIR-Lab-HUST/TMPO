# GRPO 系列方法全面对比分析

> 对比 **flow_grpo** (GRPO-Guard) / **MixGRPO** / **TreeGRPO** / **FlowRL** 四种方法
> 以及 TreeMatch-RL 如何融合各自优势

---

## 一、方法概览

| | flow_grpo (GRPO-Guard) | MixGRPO | TreeGRPO | FlowRL | **TreeMatch-RL** |
|-|----------------------|---------|----------|--------|-----------------|
| **论文** | Flow-GRPO (2024) | MixGRPO (arXiv 2507.21802) | TreeGRPO (ICLR 2026, arXiv 2512.08153) | FlowRL (arXiv 2509.15207) | *Ours* |
| **团队** | yifan123 | 腾讯混元 (Tencent Hunyuan) | Zheng Ding, Weirui Ye | Xuekai Zhu 等 | - |
| **应用领域** | 扩散模型对齐 (图像) | 扩散模型对齐 (图像) | 扩散模型对齐 (图像) | **LLM 推理** (非扩散) | 扩散模型对齐 (图像) |
| **核心思想** | PPO-clip + SDE 全步采样 | 混合 ODE-SDE 滑动窗口加速 | 树状优势估计减少方差 | GFlowNet 流平衡匹配分布 | Softmax-TB 分布匹配 + 树采样 |

---

## 二、训练数据集

| 方法 | 数据集 | Prompt 来源 | 规模 |
|------|--------|------------|------|
| **flow_grpo** | `dataset/pickscore`, `dataset/ocr`, `dataset/geneval` | Pick-a-Pic 用户偏好 / OCR 文字 / GenEval 组合 | ~2K prompts |
| **MixGRPO** | [HPDv2](https://huggingface.co/datasets/ymhao/HPDv2) 训练集 | Human Preference Dataset v2 | ~大规模 |
| **TreeGRPO** | [HPDv2](https://huggingface.co/datasets/ymhao/HPDv2) | `prompts.txt` (HPDv2 prompt 子集) | ~大规模 |
| **FlowRL** | [math_data](https://huggingface.co/xuekai/flowrl-data-collection), code_data | 数学推理 / 代码题 | 数学+代码 |
| **TreeMatch-RL** | `data/pickapic_prompts.json` | Pick-a-Pic prompt 子集 | 自定义 |

> **注意**: FlowRL 是 LLM 方法, 数据集是文本推理题; 其余都是扩散模型方法, 数据集是图像生成 prompt。

---

## 三、训练模型

| 方法 | 基座模型 | 参数量 | 微调方式 |
|------|---------|--------|---------|
| **flow_grpo** | SD3.5-Medium / FLUX.1-dev / Wan2.1 / Qwen-Image | 1.3B~12B | LoRA + FSDP |
| **MixGRPO** | **FLUX.1-dev** | ~12B | **全参数** fp32 |
| **TreeGRPO** | **SD3.5-Medium** | ~2.5B | **全参数** (LoRA 可选, 默认关闭) |
| **FlowRL** | Qwen2.5-7B/32B / DeepSeek-7B | 7B~32B | 全参数 (verl 框架) |
| **TreeMatch-RL** | SD3.5-Medium / Flux | ~2.5B/12B | **LoRA** + FSDP ZeRO-2 |

---

## 四、GPU 需求

| 方法 | 默认配置 | 最小配置 | 分布式框架 | 混合精度 |
|------|---------|---------|----------|---------|
| **flow_grpo** | **32×80GB** A100/H100 | 1×80GB | Accelerate + FSDP | bf16 |
| **MixGRPO** | **32×80GB** (4节点×8卡) | - | **torchrun + pdsh** 多节点 | fp32 (全精度!) |
| **TreeGRPO** | **8×80GB** | 8×80GB | Accelerate | bf16 |
| **FlowRL** | 多卡 (verl 框架) | - | verl + Ray | bf16 |
| **TreeMatch-RL** | **8×H200 80GB** | 1×24GB | Accelerate + FSDP ZeRO-2 | bf16 |

---

## 五、奖励函数

| 方法 | 奖励模型 | 单/多奖励 | 融合方式 |
|------|---------|---------|---------|
| **flow_grpo** | OCR / PickScore / CLIPScore / GenEval (逐任务单一) | 单一 | - |
| **GRPO-Guard** | OCR / PickScore / GenEval (逐任务单一) | 单一 | - |
| **MixGRPO** | **HPSv2 + ImageReward + PickScore + CLIPScore** | **四模型** | 多奖励加权 |
| **TreeGRPO** | HPSv2 | 单一 | - |
| **FlowRL** | 数学 ORM / 代码测试 (非图像奖励) | 单一 | - |
| **TreeMatch-RL** | **HPSv2 + CLIPScore** | **双模型** | advantage_aggr (独立归一化) |

---

## 六、训练方法对比

### 6.1 采样策略

```
flow_grpo:     z₀ ─ SDE₁ ─ SDE₂ ─ ... ─ SDE₁₀ ─ x     (K=24 条完全独立路径)
               每步都注入噪声, 24 路径各自采样

GRPO-Guard:    同 flow_grpo, 但 ratio 计算加了 RatioNorm 偏置修正

MixGRPO:       z₀ ─ ODE ─ ODE ─ [SDE窗口 3步] ─ ODE尾段 ─ x
               只在滑动窗口内注入噪声, 窗口外用 ODE
               Flash 模式: 窗口后 ODE 压缩为 DPM-Solver

TreeGRPO:      z₀ ─ ODE ── SDE分叉(×2) ─ ODE ── SDE分叉(×2) ── ...
               树结构: w=4 层, k=2 分支 → 2⁴=16 条路径
               窗口滑动: 每 tou=50 epochs 移动窗口位置

TreeMatch-RL:  z₀ ─ ODE ── SDE分叉₁(×3) ─ ODE ── SDE分叉₂(×3) ── SDE分叉₃(×3) ── DPM尾段
               树结构: 3层, k=3 → 3³=27 条路径
               自适应: Beta 分布动态决定分叉位置
```

### 6.2 损失函数

| 方法 | 损失函数 | 数学形式 | 目标 |
|------|---------|---------|------|
| **flow_grpo** | PPO-Clip + KL | `L = -adv × max(ratio, clip(ratio)) + β×KL` | 奖励最大化 |
| **GRPO-Guard** | PPO-Clip + RatioNorm | `L = PPO / sqrt_dt² + β×KL` | 奖励最大化 (修偏) |
| **MixGRPO** | PPO-Clip (极小 ε) | `L = -adv × clip(ratio, 1±ε)`, ε=1e-5 | 奖励最大化 (近似) |
| **TreeGRPO** | **树优势** PPO | `L = -tree_adv × clip(ratio)` | 方差降低的奖励最大化 |
| **FlowRL** | **GFlowNet TB Loss** | `L = (log Z + log π - β×r - log π_ref)²` | **分布匹配** |
| **TreeMatch-RL** | **Softmax-TB** + IS + Entropy + Ref | `L = w × (log_softmax(π) - log_softmax(β×R))²` | **分布匹配** |

### 6.3 训练超参对比

| 参数 | flow_grpo | GRPO-Guard | MixGRPO | TreeGRPO | **TreeMatch-RL** |
|------|----------|------------|---------|----------|-----------------|
| lr | 1e-5 | 3e-5 | (未公开) | 1e-5 | 1e-5 |
| 采样步数 | 10 | 10 | 6 (Flux) | 10 | **28** |
| 路径数/prompt | 24 | 24 | 24 | k^w=16 (k=2,w=4) | **27** (k=3,层=3) |
| KL β | 0.01~0.04 | 0.0 | 0.0 | 0.0 | 15.0 (TB 温度) |
| clip_range ε | 0.2 | 0.2 | 1e-5 | 1e-4 | 0.2 |
| noise η | 0.7 | 0.7 | 0.8 | 0.7 | [0.4, 0.7, 1.0] |
| IS 更新次数 | 4 | 4 | 1 (单次) | 1 | 4 (动态对齐) |
| grad_accum | 1~2 | 1~2 | 1 | 8 | 3 |
| EMA | ✅ | ✅ | ✅ | ❌ | ❌ |
| CFG | 4.5 | 4.5 | 3.5 | 4.5 | ❌ |

---

## 七、核心方法论差异

### 7.1 flow_grpo — 基线方法

**核心**: 标准 PPO-Clip 应用于扩散模型。全步 SDE 采样 K 条独立路径, 计算优势, PPO 策略梯度更新。

- ✅ 简单直接, 收敛稳定
- ❌ K 路径完全独立, 无前缀共享 → 计算冗余大
- ❌ IS ratio 有 E[log w] < 0 偏置
- ❌ 只最大化奖励, 容易模式坍缩

### 7.2 GRPO-Guard — 偏置修正

**核心**: 在 flow_grpo 基础上加入 **RatioNorm**, 解决 IS ratio 的均值偏置问题。

```python
# GRPO-Guard 的 RatioNorm
bias = ||μ_θ - μ_old||² / (2 × (√dt × σ_t)²)
ratio = exp((log_π - log_π_old + bias) × √dt × σ_t)
loss /= √dt²
```

- ✅ 消除了 IS ratio 的期望偏置
- ✅ 支持多次策略更新 (is_num_updates=4)
- ❌ 仍然是独立采样, 无树结构

### 7.3 MixGRPO — ODE-SDE 混合加速

**核心**: 不是全步 SDE, 而是**只在部分步用 SDE** (滑动窗口), 窗口外用 ODE。大幅减少需要反传梯度的步数。

关键创新:
1. **SDE 窗口** — 在 6 步中只有 3 步是 SDE, 其余 ODE
2. **Flash 模式** — 窗口后 ODE 用 DPM-Solver 压缩步数
3. **CPS 采样** — Coefficients-Preserving Sampling, 更原则性的 SDE 替代
4. **多奖励** — HPSv2 + ImageReward + PickScore + CLIPScore 四模型

- ✅ **NFE 大幅减少** (从 10 步降到 2-3 步训练)
- ✅ **多奖励融合** 效果更好
- ❌ 需要 32 卡全参数训练 (资源门槛极高)
- ❌ clip_range=1e-5 约等于不 clip (近似 REINFORCE)

### 7.4 TreeGRPO — 树状优势降方差

**核心**: 用**树结构**共享前缀, 在树的每个分叉点计算局部优势, 降低方差。

关键参数:
- `w=4` — 树深度 4 层
- `k=2` — 每层 2 分支 → 2⁴=16 路径
- `tou=50` — 每 50 epochs 移动窗口
- `s=1` — 窗口每次移动 1 步

```
树优势 = 在同一个分叉点的兄弟节点间归一化
→ 共同前缀取消 → 方差更低
```

- ✅ 前缀共享降低计算
- ✅ 树优势降低方差 → 更稳定训练
- ❌ 窗口是固定的 (手动调 tou 和 s)
- ❌ 无多样性机制

### 7.5 FlowRL — LLM 上的分布匹配

**核心**: 将 **GFlowNet 的流平衡** (Flow Balance) 引入 LLM RL 训练, 目标是匹配分布而非最大化奖励。

$$L = w \cdot (log Z + \frac{1}{|y|} log \pi_\theta - \beta \hat{r} - \frac{1}{|y|} log \pi_{ref})^2$$

- ✅ **分布匹配** → 多样化推理路径
- ✅ 数学/代码推理效果显著
- ❌ 是 LLM 方法, 不直接适用于扩散模型
- ❌ 依赖 verl 框架

---

## 八、TreeMatch-RL 如何融合优势

| 借鉴来源 | 借鉴的组件 | TreeMatch-RL 中的实现 |
|---------|----------|---------------------|
| **TreeGRPO** | 树状采样结构 | 3 层 k=3 树, 27 路径 (vs TreeGRPO 的 k=2, 16 路径) |
| **FlowRL** | GFlowNet 分布匹配 | **Softmax-TB** 损失 (从 LLM 适配到扩散模型) |
| **MixGRPO** | DPM Flash 加速 | ODE 尾段 DPM-Solver++ 压缩 |
| **GRPO-Guard** | RatioNorm IS 修偏 | RatioNorm 标准化 IS 权重 |
| **MixGRPO** | 多奖励融合 | advantage_aggr (独立归一化后加权) |
| *原创* | Beta 分布自适应 | 根据在线奖励动态调分叉位置 |
| *原创* | RBF 粒子熵正则 | 防止 27 路径坍缩为同一解 |
| *原创* | 低 GPU 适配 | LoRA + FSDP ZeRO-2, 最低 1×24GB |
