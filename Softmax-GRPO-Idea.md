# Softmax-GRPO: 基于分布匹配的树状扩散模型对齐

> **项目名称**: Softmax-GRPO (原 Z-AFT)
> **核心定位**: 将 GFlowNet 的轨迹平衡 (TB) 思想与扩散模型的树状采样结构深度融合，通过 Softmax 分布匹配替代传统奖励最大化，实现高质量且多样性兼顾的扩散模型对齐。

---

## 一、三个分支点噪声系数设计

### 1.1 噪声系数的数学原理

在 Flow Matching 框架下，ODE→SDE 转换的核心公式为（来自 flow_grpo 和 TreeGRPO 共同使用的 `sde_step_with_logprob`）：

$$\sigma_t = \sqrt{\frac{\sigma}{1 - \sigma}} \cdot \eta$$

其中 $\sigma$ 是当前时间步的信噪比参数，$\eta$ 是 **噪声系数 `noise_level`**。

SDE 步进的均值和标准差为：

$$\mu = x_t \left(1 + \frac{\sigma_t^2}{2\sigma} dt \right) + v_\theta \left(1 + \frac{\sigma_t^2(1-\sigma)}{2\sigma} \right) dt$$

$$\text{std} = \sigma_t \sqrt{-dt}$$

$$x_{t-1} = \mu + \text{std} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**噪声系数 $\eta$ 的物理意义**：控制 SDE 注入噪声的强度，直接决定了分支间的差异程度——$\eta$ 越大，分支间的多样性越强，但稳定性越差。

---

### 1.2 参考项目的噪声系数策略

| 项目 | 噪声系数 $\eta$ | SDE 应用范围 | 分叉策略 |
|------|----------------|------------|---------|
| **flow_grpo** | 固定 `noise_level=0.7`（全局） | SDE Window：仅 2 个步注入噪声 | 无树结构，独立 K-Repeat 采样 |
| **TreeGRPO** | 固定 `noise_level=0.7`（全局） | 仅在 `tree_steps` 使用 SDE，其余 ODE | 每步 k=2 分叉，w=4 步，共 16 分支 |
| **FlowRL** | N/A（LLM 领域，无扩散噪声） | token-level log_prob | N/A |

> [!IMPORTANT]
> **关键发现**：flow_grpo 和 TreeGRPO **均使用固定全局 `noise_level=0.7`**，未对不同分叉层使用不同系数。这为 Softmax-GRPO 提供了差异化创新空间。

---

### 1.3 Softmax-GRPO 的三阶噪声系数设计

采用 **10 步去噪流程**（`num_inference_steps=10`），在三个关键时间步进行 SDE 分叉，每步分 3 个分支，最终形成 $3^3 = 27$ 条路径。

#### 方案 A：递增噪声（推荐 — 对应 DenseGRPO 的时变噪声思想）

| 分叉层 | 时间步索引 | $\sigma$ 范围 | 噪声系数 $\eta$ | 设计理由 |
|-------|-----------|--------------|----------------|---------|
| 第 1 层 | step=2（$t \approx 0.8$） | 高噪声区 | $\eta_1 = 0.4$ | 早期 latent 信息量低，高噪声不携带有效梯度信号，小噪声保持结构一致性 |
| 第 2 层 | step=5（$t \approx 0.5$） | 中噪声区 | $\eta_2 = 0.7$ | 中期是语义分化的关键阶段，标准噪声提供足够多样性 |
| 第 3 层 | step=8（$t \approx 0.2$） | 低噪声区 | $\eta_3 = 1.0$ | 后期 latent 已接近清晰图像，需要更大噪声才能产生可区分的细节变化 |

**数学直觉**：$\sigma_t$ 的公式中 $\sigma/(1-\sigma)$ 在 $t$ 较大时本身已经很大（早期噪声多），此时 $\eta$ 应较小以避免发散；$t$ 较小时 $\sigma/(1-\sigma)$ 很小（后期信号干净），此时 $\eta$ 应增大以产生有意义的分支差异。

```python
# 噪声系数配置
tree:
  split_steps: [2, 5, 8]         # 三个分叉时间步索引
  noise_levels: [0.4, 0.7, 1.0]  # 对应的递增噪声系数
  k: 3                           # 每步 3 分支
  use_ode: true                  # 非分叉步使用 ODE
```

#### 方案 B：固定噪声（对照基线）

统一使用 `noise_level=0.7`，与 TreeGRPO/flow_grpo 保持一致，作为消融实验基线。

```python
tree:
  split_steps: [2, 5, 8]
  noise_levels: [0.7, 0.7, 0.7]  # 固定值
  k: 3
  use_ode: true
```

#### 方案 C：Z-自适应噪声（进阶实验）

根据 DenseGRPO 提出的 **奖励感知探索校准** 思想，结合在线估计的难度 $\bar{R}$ 动态调整：

```python
def adaptive_noise_levels(z_estimate, base_levels=[0.4, 0.7, 1.0]):
    """根据在线估计的 Z (难度) 自适应调整噪声系数"""
    if z_estimate < z_low:   # 难题
        scale = 1.3          # 增大噪声，加强探索
    elif z_estimate > z_high: # 简单题
        scale = 0.7          # 减小噪声，节省算力
    else:
        scale = 1.0
    return [min(η * scale, 1.5) for η in base_levels]
```

---

### 1.4 噪声系数对 log_prob 的影响

从源码中的 log_prob 计算公式：

$$\log p(x_{t-1} | x_t) = -\frac{(x_{t-1} - \mu)^2}{2 \cdot \text{std}^2} - \log(\text{std}) - \frac{1}{2}\log(2\pi)$$

**$\eta$ 越大 → $\text{std}$ 越大 → log_prob 的绝对值越小 → 分支间的概率差异越平滑。** 这意味着在 Soft-TB 损失中，大噪声的分叉层对概率分布匹配的影响较弱，而小噪声的层影响更强。因此需要在设计 Soft-TB 时对不同层做 **长度归一化**（参考 FlowRL 和 updata.md 中的建议）。

---

## 二、Reward 设计方案

### 2.1 三个参考项目的 Reward 策略对比

| 维度 | flow_grpo | TreeGRPO | FlowRL | DenseGRPO |
|------|-----------|----------|--------|-----------|
| **奖励粒度** | 终端奖励（稀疏） | 终端奖励（稀疏） | 终端奖励（稀疏） | **逐步密集奖励** |
| **奖励类型** | 12 种可组合（HPSv2, PickScore, CLIP, Aesthetic, ImageReward, OCR, GenEval 等） | 仅 HPSv2 | 数学/代码正确性 (0/1) | 任意终端奖励 + ODE 预测的步级增益 |
| **奖励归一化** | Per-Prompt 均值/方差归一化 或 全局归一化 | 全局均值/方差归一化 | β=15 缩放 | 步级增益自然归一化 |
| **多样性保障** | Per-Prompt 统计 + KL 正则 | 树结构分叉 | TB 分布匹配（固有保障） | reward-aware 探索校准 |
| **优势估计** | `(R - mean) / std` 每步共享 | **树状递归**：叶子→内部节点平均传播 | 不使用优势（TB 残差替代） | 逐步奖励增益替代统一优势 |

---

### 2.2 Softmax-GRPO 的 Reward 设计

#### 核心思路：融合三方优势

1. **从 DenseGRPO 借鉴**：逐步密集奖励 → 为每个分叉步提供独立的梯度信号
2. **从 TreeGRPO 借鉴**：树状优势回传 → 结构化信用分配
3. **从 flow_grpo 借鉴**：多奖励组合 + Per-Prompt 归一化 → 多任务泛化
4. **从 FlowRL 借鉴**：分布匹配 (Soft-TB) → 多样性天然保持

---

#### 2.2.1 终端奖励：多奖励加权组合

直接复用 flow_grpo 的 `multi_score` 架构：

```python
reward_config = {
    "hpsv2": 0.5,           # 人类偏好评分
    "aesthetic": 0.3,       # 美学评分
    "clipscore": 0.2,       # 文图一致性
}
# 加权总奖励
R_terminal = Σ w_i * R_i
```

**针对不同实验任务的配置**：

| 实验场景 | 奖励组合 | 权重 |
|---------|---------|------|
| 通用美学提升 | HPSv2 + Aesthetic | 0.6 + 0.4 |
| 文图一致性 | GenEval (strict) | 1.0 |
| 文字渲染 | OCR + Aesthetic | 0.7 + 0.3 |
| 综合优化 | HPSv2 + Aesthetic + CLIP | 0.5 + 0.3 + 0.2 |

---

#### 2.2.2 密集奖励（DenseGRPO 启发）：ODE 预测的分叉步奖励增益

> [!NOTE]
> DenseGRPO 的核心创新：在每个去噪步，通过 ODE 将中间 latent 投射到 $t=0$，用终端奖励模型对这个"预测清晰图像"打分，从而获得逐步的奖励增益。

**适配到三阶树结构**：

```python
def compute_dense_rewards(tree_root, reward_fn, pipeline):
    """为每个分叉步计算密集奖励增益"""
    
    for split_node in tree_root.get_split_nodes():
        # 1. 用 ODE 将分叉前 latent 投射到 t=0
        x0_before = ode_predict_clean(pipeline, split_node.latent_in, split_node.timestep)
        R_before = reward_fn(decode(x0_before))
        
        # 2. 对每个子分支，用 ODE 将分叉后 latent 投射到 t=0
        for child in split_node.children:
            x0_after = ode_predict_clean(pipeline, child.latent_out, next_timestep)
            R_after = reward_fn(decode(x0_after))
            
            # 3. 步级奖励增益 = 分叉后 - 分叉前
            child.step_reward_gain = R_after - R_before
    
    return tree_root
```

**ODE 投射函数**（参考 DenseGRPO 的方式，利用 Flow Matching 的线性性质）：

$$\hat{x}_0 = x_t - \sigma_t \cdot v_\theta(x_t, t)$$

这在 flow_grpo 的 CPS 模式中已有实现（见 `sd3_sde_with_logprob.py` 第 72 行）。

---

#### 2.2.3 综合奖励策略：融合终端稀疏奖励 + 分叉步密集奖励

```
最终每条路径的奖励信号 = 终端奖励 R_terminal + α * Σ(步级奖励增益)
```

在 Soft-TB 损失中的使用方式：

```python
class SoftTBWithDenseReward(nn.Module):
    """融合密集奖励的 Softmax-TB 损失"""
    
    def forward(self, path_log_probs, terminal_rewards, 
                step_reward_gains, alpha=0.3):
        """
        Args:
            path_log_probs:     (27,) 每条路径的累积 log_prob
            terminal_rewards:   (27,) 终端奖励
            step_reward_gains:  (27, 3) 每条路径在 3 个分叉步的奖励增益
            alpha:              密集奖励权重
        """
        # 综合奖励
        dense_bonus = step_reward_gains.sum(dim=1)  # (27,)
        combined_reward = terminal_rewards + alpha * dense_bonus
        
        # Softmax 分布匹配
        log_p = F.log_softmax(path_log_probs, dim=0)
        log_r = F.log_softmax(combined_reward, dim=0)
        
        # TB 残差
        loss = ((log_p - log_r) ** 2).sum()
        return loss
```

---

#### 2.2.4 树状优势估计（TreeGRPO 启发 + 密集奖励增强版）

> [!NOTE]
> TreeGRPO 的树状优势：叶节点 advantage = `(R - mean) / std`，内部节点 advantage = 子节点 advantage 的平均值。这确保每个分叉步都能获得来自所有后续分支的梯度信号。

**Softmax-GRPO 的增强版本**：为内部节点加上密集奖励增益信号。

```python
def compute_tree_advantage(node, global_mean, global_std, use_dense=True):
    """递归计算树状优势（增强版）"""
    if node.is_leaf():
        # 叶节点：标准归一化优势
        node.advantage = [(node.reward - global_mean) / global_std]
        return node.advantage[0]
    else:
        child_advs = []
        for child in node.children:
            adv = compute_tree_advantage(child, global_mean, global_std, use_dense)
            child_advs.append(adv)
        
        # 内部节点：子节点平均 + 密集奖励增益
        avg_child_adv = torch.stack(child_advs).mean()
        
        if use_dense and hasattr(node, 'step_reward_gain'):
            # 融合该步的密集奖励信号
            dense_signal = node.step_reward_gain / global_std
            node.advantage = [avg_child_adv + 0.1 * dense_signal 
                              for _ in node.children]
        else:
            node.advantage = [avg_child_adv for _ in node.children]
        
        return avg_child_adv
```

---

#### 2.2.5 奖励感知的探索校准（DenseGRPO 思想的延伸）

DenseGRPO 指出：**统一的噪声设置与时变的噪声强度之间存在不匹配**。Softmax-GRPO 通过递增噪声系数 $\eta_1 < \eta_2 < \eta_3$ 解决了这个问题。

进一步，结合密集奖励增益，可以做 **步级自适应探索**：

```python
def reward_aware_noise_calibration(step_reward_gains, base_noise_levels):
    """根据步级奖励增益方差自适应调整噪声
    
    原理：如果某一步的奖励增益方差很小 → 说明该步对质量影响不大 → 可减小噪声
           如果方差很大 → 说明该步是关键决策点 → 应增大噪声加强探索
    """
    for i, (gain_var, base_η) in enumerate(zip(
        step_reward_gains.var(dim=0),  # 每步的奖励增益方差
        base_noise_levels
    )):
        if gain_var > threshold_high:
            adjusted_η[i] = min(base_η * 1.2, 1.5)  # 关键步加强探索
        elif gain_var < threshold_low:
            adjusted_η[i] = max(base_η * 0.8, 0.2)  # 非关键步减少噪声
        else:
            adjusted_η[i] = base_η
    return adjusted_η
```

---

## 三、总体 Loss 函数

将以上设计整合为完整的损失函数：

$$\mathcal{L}_{total} = w \cdot \mathcal{L}_{Soft\text{-}TB} + \lambda_1 \mathcal{L}_{Entropy} + \lambda_2 \mathcal{L}_{Ref}$$

| 损失项 | 公式 | 来源 | 功能 |
|--------|------|------|------|
| **Soft-TB** | $\sum_{i=1}^{27}\left(\log\frac{P_\theta(\tau_i)}{\sum_j P_\theta(\tau_j)} - \log\frac{R_i}{\sum_j R_j}\right)^2$ | FlowRL TB → 改造 | 路径概率与综合奖励的分布匹配 |
| **粒子熵** | $\frac{1}{K(K-1)}\sum_{i\neq j}\exp\left(-\frac{\|\phi_i - \phi_j\|^2}{h}\right)$ | 原创 | RBF 核防止 27 分支语义坍缩 |
| **参考约束** | $\sum_{i=1}^{27}\left(\frac{\log\pi_\theta(\tau_i)}{|\tau|} - \frac{\log\pi_{ref}(\tau_i)}{|\tau|}\right)^2$ | FlowRL ref + flow_grpo KL | 防止策略偏离预训练分布 |
| **IS 权重** | $w = \text{clip}\left(\frac{\pi_\theta(\tau)}{\pi_{old}(\tau)}, 1-\epsilon, 1+\epsilon\right)^{\text{detach}}$ | FlowRL IS | 支持一次采样 4-8 次更新 |

**超参数推荐值**：

```yaml
loss:
  lambda_1: 0.01       # 粒子熵权重（过大会牺牲质量，过小失去多样性）
  lambda_2: 0.1        # 参考约束权重
  alpha: 0.3           # 密集奖励增益权重
  is_clip_range: 0.2   # IS 裁剪范围
  num_updates: 4       # 每次采样后的更新次数
```

---

## 四、消融实验设计

| 实验 | 变量 | 评估指标 |
|------|------|---------|
| **噪声系数方案** | 固定 0.7 vs 递增 [0.4, 0.7, 1.0] vs Z-自适应 | HPSv2 ↑, FID ↓, LPIPS 多样性 ↑ |
| **密集奖励** | 无 vs ODE 预测增益 vs Ground Truth 增益 | 收敛速度, 最终 HPSv2 |
| **Soft-TB vs PPO-Clip** | 分布匹配 vs 奖励最大化 | HPSv2 ↑, Vendi Score (多样性) ↑ |
| **粒子熵** | 无 vs CLIP 空间 vs Latent 空间 | LPIPS diversity, 视觉检查 |
| **分叉层数** | 2 层 (9 路径) vs 3 层 (27 路径) vs 4 层 (81 路径) | HPSv2, 显存, 训练速度 |

---

## 五、与竞品的核心差异总结

```
Softmax-GRPO = TreeGRPO 的树结构 + FlowRL 的分布匹配 + DenseGRPO 的密集奖励
```

| | Softmax-GRPO | TreeGRPO | flow_grpo | DenseGRPO |
|--|---|---|---|---|
| 损失目标 | **分布匹配** (Soft-TB) | 奖励最大化 (PPO) | 奖励最大化 (PPO + KL) | 奖励最大化 (PPO) |
| 采样结构 | 3阶27分支树 | w步2分叉树 | 无树 (K-Repeat) | 无树 (独立采样) |
| 噪声系数 | **递增 (0.4→0.7→1.0)** | 固定 0.7 | 固定 0.7 | **奖励感知自适应** |
| 奖励粒度 | **终端 + 密集 (融合)** | 终端 | 终端 | **密集 (核心)** |
| 多样性机制 | **Soft-TB + RBF 粒子熵** | 树分叉 | Per-Prompt 统计 | 无显式机制 |
| Z 处理 | softmax 消除 | 无 | 无 | 无 |
| 多次更新 | IS (4-8次) | 1次 | 1次 | 1次 |
