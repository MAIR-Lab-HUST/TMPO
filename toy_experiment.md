# TreeMatch-RL vs Flow-GRPO 实验设计：粒子分布匹配验证

## 1. 实验动机与目标

### 1.1 核心假说

TreeMatch-RL 的 Softmax-TB 损失以**奖励分布匹配**（$\pi \propto \exp(\beta R)$）为目标，而 Flow-GRPO 以**奖励最大化**（$\max \mathbb{E}[R]$）为目标。我们预期：

> **在多模态奖励景观下，TreeMatch-RL 能更好地覆盖所有奖励峰值，而 Flow-GRPO 会坍缩到少数高奖励模态。**

### 1.2 为何选用粒子分布匹配实验

扩散模型的完整训练代价高昂，因此我们设计一个**低维度的合成实验**来快速验证核心机理：

- **可控性**：目标分布已知，可精确度量匹配质量
- **可视化**：2D 粒子可直接绘图展示分布覆盖
- **速度**：几分钟内在单 GPU 或 CPU 上即可跑完
- **说服力**：直观展示模式塌缩 vs 分布匹配的差异

---

## 2. 实验设定

### 2.1 问题定义：2D 粒子匹配多模态奖励分布

设计一个连续空间的生成任务：

```
输入: z ~ N(0, I)       （2D 高斯噪声）
生成: x = G_θ(z)        （可学习的生成网络）
奖励: R(x) = 高斯混合   （多模态目标分布）
```

**目标分布**（奖励函数）定义为 **5 个 2D 高斯分布**的混合：

$$R(x) = \sum_{m=1}^{5} w_m \cdot \mathcal{N}(x \mid \mu_m, \sigma_m^2 I)$$

| 模态 | 中心 $\mu_m$ | 权重 $w_m$ | 标准差 $\sigma_m$ |
|------|-------------|-----------|-----------------|
| A | $(3, 3)$ | 0.30 | 0.5 |
| B | $(-3, 3)$ | 0.25 | 0.5 |
| C | $(-3, -3)$ | 0.20 | 0.5 |
| D | $(3, -3)$ | 0.15 | 0.5 |
| E | $(0, 0)$ | 0.10 | 0.8 |

各模态权重不同，形成一个**非均匀**的多模态分布，以测试方法能否按比例覆盖。

### 2.2 生成模型："Mini-Diffusion" 通过 N 步去噪

为模拟扩散模型的逐步去噪过程，构建一个**简化的 N 步流匹配模型**：

```python
class MiniFlowModel(nn.Module):
    """简化版流匹配模型：N 步线性插值去噪"""
    def __init__(self, hidden_dim=64, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        # 速度场网络 v_θ(x_t, t)
        self.velocity_net = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim),    # 输入: x_t (2D) + 时间步 t (1D)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),          # 输出: 速度向量 (2D)
        )

    def forward(self, x_t, t):
        """预测速度场 v_θ(x_t, t)"""
        t_emb = t.unsqueeze(-1) if t.dim() == 0 else t
        inp = torch.cat([x_t, t_emb.expand(x_t.shape[0], 1)], dim=-1)
        return self.velocity_net(inp)
```

**采样过程**（ODE/SDE）：

```python
def sample_ode(model, z, num_steps=10):
    """ODE 采样：z → x_0"""
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
    x = z
    for i in range(num_steps):
        t = sigmas[i]
        v = model(x, t)
        dt = sigmas[i+1] - sigmas[i]
        x = x + dt * v
    return x

def sample_sde(model, z, noise_level=0.5, num_steps=10):
    """SDE 采样（注入噪声，计算 log_prob）"""
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
    x = z
    total_log_prob = 0.0
    for i in range(num_steps):
        t = sigmas[i]
        v = model(x, t)
        dt = sigmas[i+1] - sigmas[i]
        mean = x + dt * v
        std = noise_level * abs(dt) ** 0.5
        noise = torch.randn_like(x)
        x = mean + std * noise
        # log probability
        log_prob = -0.5 * (noise ** 2).sum(-1) - x.shape[-1] * 0.5 * math.log(2 * math.pi)
        total_log_prob += log_prob
    return x, total_log_prob
```

### 2.3 树状采样（TreeMatch-RL 侧）

在去噪的 $N=10$ 步中选 **2 个分叉点**（简化为 2 阶 $3^2=9$ 路径）：

```python
# 每条路径在分叉步注入独立 SDE 噪声
# root → ODE → SDE_split_1 (×3) → ODE → SDE_split_2 (×3) → ODE → 9 leaves
split_steps = [3, 7]      # 在第 3 步和第 7 步分叉
noise_levels = [0.5, 0.3]  # 分叉步噪声强度
```

### 2.4 平行采样（Flow-GRPO 侧）

每次独立采样 **K=9 条**平行路径（与 TreeMatch 相同数量），全程使用 SDE 采样。

---

## 3. 对比的训练方法

### 3.1 Method A: TreeMatch-RL（Softmax-TB 分布匹配）

```python
# ═══ 算法核心 ═══
# 1. 树状采样 → 9 条路径，各有 log_prob 和 reward
# 2. Softmax-TB 损失：匹配 路径概率分布 与 奖励分布

def softmax_tb_loss(path_log_probs, rewards, beta=15.0):
    """Softmax-TB: 无需配分函数 Z"""
    log_p = F.log_softmax(path_log_probs, dim=0)     # 路径概率归一化
    log_r = F.log_softmax(beta * rewards, dim=0)      # 奖励分布归一化
    return ((log_p - log_r) ** 2).sum()
```

**关键特性**：
- 目标是让路径概率 $\propto \exp(\beta R)$，按比例覆盖所有模态
- 通过组内归一化消除了配分函数 $Z$
- $\beta$ 控制多样性-质量权衡

### 3.2 Method B: Flow-GRPO（奖励最大化）

```python
# ═══ 算法核心 ═══
# 1. 独立采样 → 9 条路径
# 2. 计算 GRPO 优势并做 PPO-style 策略梯度

def grpo_loss(log_probs, old_log_probs, advantages, clip_range=0.2):
    """GRPO: 奖励最大化"""
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    return torch.maximum(unclipped, clipped).mean()

# 优势计算: 组内归一化
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
```

**关键特性**：
- 目标是最大化期望奖励 $\mathbb{E}[R]$
- 通过 PPO-clip 防止过大步长
- 没有分布匹配约束，容易趋向单一高奖励模态

### 3.3 Method C: Baseline（预训练模型 / 无 RL）

直接使用未微调的生成模型，作为基准对照。

---

## 4. 评估指标

### 4.1 定量指标

| 指标 | 计算方式 | 衡量内容 |
|------|----------|---------|
| **平均奖励** $\bar{R}$ | $\frac{1}{N}\sum R(x_i)$ | 生成质量 |
| **模态覆盖率** (Mode Coverage) | 覆盖到的模态数 / 总模态数 | 多样性 |
| **KL 散度** $D_{KL}(p_{gen} \| p_{target})$ | 生成分布与目标分布的 KL | 分布匹配精度 |
| **Jensen-Shannon 散度** $D_{JS}$ | $\frac{1}{2}D_{KL}(p\|m) + \frac{1}{2}D_{KL}(q\|m)$ | 对称分布距离 |
| **每模态比例误差** | $|\hat{w}_m - w_m|$ | 权重匹配精度 |
| **样本多样性** (Entropy / Vendi Score) | 生成样本间的多样性 | 避免模式塌缩 |

### 4.2 定性可视化

1. **2D 散点图**：生成粒子 + 目标分布等高线叠加
2. **训练曲线**：平均奖励、模态覆盖率 vs 训练步数
3. **分布直方图**：各模态的实际生成频率 vs 目标比例
4. **动态 GIF**：训练过程中粒子分布的演化

---

## 5. 实验参数设置

### 5.1 通用参数

```python
config = {
    # 模型
    "hidden_dim": 64,
    "num_steps": 10,              # 去噪步数
    "noise_level": 0.5,           # SDE 噪声级别

    # 训练
    "learning_rate": 1e-3,
    "num_epochs": 500,
    "batch_size_per_sample": 32,  # 每次采样的初始噪声个数(prompt维度)
    "group_size": 9,              # 每组路径数 (K=9, 3²)

    # 评估
    "eval_samples": 2000,         # 评估用粒子数
    "eval_interval": 50,          # 每 50 epoch 评估
}
```

### 5.2 TreeMatch-RL 特有参数

```python
treematch_config = {
    "beta": 15.0,                 # 温度参数
    "lambda_entropy": 0.01,       # 粒子熵系数
    "lambda_ref": 0.1,            # 参考约束系数
    "split_steps": [3, 7],        # 分叉步
    "noise_levels": [0.5, 0.3],   # 分叉噪声
    "is_clip_range": 0.2,         # IS 裁剪范围
}
```

### 5.3 Flow-GRPO 特有参数

```python
grpo_config = {
    "clip_range": 0.2,            # PPO clip
    "adv_clip_max": 5.0,          # 优势裁剪
    "beta_kl": 0.01,              # KL 正则系数 (可选)
}
```

---

## 6. 实验步骤

### Phase 1: 环境搭建（0.5 天）

```bash
# 1. 创建实验目录
mkdir -p experiments/particle_matching

# 2. 实现基础组件
#    - reward_fn.py: 高斯混合奖励函数
#    - mini_flow.py: 简化版流匹配模型
#    - tree_sampler_2d.py: 2D 树状采样器
#    - grpo_trainer.py: Flow-GRPO 训练器
#    - treematch_trainer.py: TreeMatch-RL 训练器
#    - eval_metrics.py: 评估指标计算
#    - visualize.py: 可视化工具
```

### Phase 2: 核心实验（1 天）

```python
# ═══ 实验 1: 基础对比 ═══
# 在标准 5 模态设定下跑 500 epochs

# ═══ 实验 2: β 消融 ═══
# TreeMatch-RL: β ∈ {1, 5, 10, 15, 25, 50}
# 验证温度参数对分布匹配的影响

# ═══ 实验 3: 模态数量扩展 ═══
# 目标分布模态数: {2, 5, 10, 20}
# 观察随模态增加，两种方法的覆盖差异

# ═══ 实验 4: 非均匀权重极端场景 ═══
# 设置极端权重: [0.6, 0.2, 0.1, 0.05, 0.05]
# 验证 TreeMatch-RL 能否按比例精确匹配
```

### Phase 3: 分析与可视化（0.5 天）

生成以下关键图表用于论文：

1. **Figure: 粒子分布对比图** (2×3 网格)
   - 行1: TreeMatch-RL / Flow-GRPO / Target
   - 行2: 对应的密度热力图

2. **Figure: 训练监控曲线** (1×3)
   - 平均奖励 vs 步数
   - 模态覆盖率 vs 步数
   - JS 散度 vs 步数

3. **Figure: β 消融热力图**
   - x 轴: β 值，y 轴: 各指标

4. **Table: 最终指标对比**

---

## 7. 预期结果

### 7.1 TreeMatch-RL 预期优势

| 指标 | TreeMatch-RL | Flow-GRPO |
|------|-------------|-----------|
| 模态覆盖率 | **5/5 (100%)** | 2-3/5 (40-60%) |
| JS 散度 ↓ | **低** | 高 |
| 模态权重误差 ↓ | **低** | 高（倾斜） |
| 平均奖励 | 适中 | **高**(但不代表真正好) |

### 7.2 直觉解释

- **Flow-GRPO** 追求奖励最大化，会把大部分粒子堆到模态 A (权重最大)，忽略 D 和 E
- **TreeMatch-RL** 要求 $\pi \propto \exp(\beta R)$，会按比例分配粒子到所有 5 个模态
- TreeMatch-RL 的平均奖励可能略低于 GRPO，但其分布形状更接近目标

---

## 8. 代码组织结构

```
experiments/particle_matching/
├── config.py               # 所有超参数配置
├── reward_fn.py             # 高斯混合奖励函数
├── mini_flow.py             # 简化版 N 步流匹配模型
├── samplers/
│   ├── tree_sampler.py      # TreeMatch 树状 SDE 采样
│   └── flat_sampler.py      # Flow-GRPO 平行 SDE 采样
├── trainers/
│   ├── treematch_trainer.py # Softmax-TB 训练循环
│   └── grpo_trainer.py      # GRPO 训练循环
├── eval_metrics.py          # KL, JS, 模态覆盖率等
├── visualize.py             # 散点图、曲线、热力图
├── run_experiment.py        # 主入口: 跑全部实验
└── README.md                # 实验说明
```

---

## 9. 与论文的关联

本实验直接验证了论文 **§1 Introduction** 和 **§4.1 Softmax-TB** 的核心论断：

> "TreeMatch-RL 旨在实现'奖励分布匹配'，即强制模型生成的路径概率与该路径获得的指数化奖励成正比 ($\pi \propto \exp(\beta R)$)。"

具体地：

- **Softmax-TB 损失** (§4.1 公式 11) 在 2D 设定中可直接可视化其优化效果
- **树状采样** (§4.2) 的前缀共享机制在 2D 同样适用且可视化分叉过程
- **β 消融** 对应论文中温度参数对多样性-质量权衡的讨论
- 结果可直接放入论文 Appendix 或 §5 Experiments 作为 "Synthetic Experiment" 小节

---

## 10. 补充说明

### 10.1 与扩散模型实验的互补性

| | 粒子匹配 (Synthetic) | 扩散模型 (SD3.5/Flux) |
|---|---|---|
| **验证范围** | 核心算法 (Softmax-TB) | 全系统 (含工程优化) |
| **可视化** | 直观 2D 散点 | 图像需人工评估 |
| **时间成本** | 分钟级 | 天/周级 |
| **论文位置** | Appendix / 辅助验证 | 主实验 |

### 10.2 可选扩展

1. **高维验证 (8D~64D)**：增加维度，观察两方法的可扩展性
2. **动态奖励景观**：训练过程中改变奖励分布，测试自适应能力
3. **与 DAG 方法对比**：加入细致平衡 (DB) 方法作为额外 baseline
4. **收敛速度分析**：绘制 "到达 $X\%$ 模态覆盖率所需步数" 的对比图
