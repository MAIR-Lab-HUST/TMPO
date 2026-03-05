# TreeMatch-RL 代码审核报告：Draft vs 实现一致性

> 本报告逐一对比 [TreeMatch_darft_01.tex](file:///Users/chaelchael/TreeMatch-RL/TreeMatch_darft_01.tex) 中的核心公式/设计与 `treematch-rl/` 代码实现，标注一致、偏差和需关注的问题。

---

## 总结

| 组件 | 一致性 | 严重程度 |
|:-----|:------|:--------|
| Softmax-TB 损失 | ✅ 一致 | — |
| 树状采样结构 (3阶27分支) | ✅ 一致 | — |
| Beta 分布自适应调度 | ✅ 一致 | — |
| 粒子熵 (RBF) | ⚠️ 公式分母偏差 | 低 |
| 参考约束 L_Ref | ✅ 一致 | — |
| RatioNorm IS | ✅ 已修复 | — |
| 总损失整合 | ✅ 一致 | — |
| DPM-Solver++ | ✅ 一致 | — |
| SDE 步进 & log_prob | ✅ 一致 | — |
| 训练循环 | ⚠️ ref_log_probs 逻辑 | 中 |

---

## 1. Softmax-TB 损失 ✅

**Draft 公式 (Eq. 12)**:
$$\mathcal{L}_{Soft\text{-}TB} = \sum_{i=1}^K \left( \log \frac{P_\theta(\tau_i)}{\sum_j P_\theta(\tau_j)} - \log \frac{\exp(\beta R_i)}{\sum_j \exp(\beta R_j)} \right)^2$$

**代码** ([softmax_tb.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/losses/softmax_tb.py#L29-L53)):
```python
log_p_normalized = F.log_softmax(path_log_probs, dim=0)
log_r_normalized = F.log_softmax(self.beta * rewards, dim=0)
residuals = log_p_normalized - log_r_normalized
loss = (residuals ** 2).sum()
```

**评判**: ✅ **完全一致**。`F.log_softmax(path_log_probs)` 等价于 [log(P_θ(τ_i) / Σ P_θ(τ_j))](file:///Users/chaelchael/TreeMatch-RL/flow_grpo-main/scripts/train_sd3_GRPO_Guard.py#181-218)，`F.log_softmax(β·R)` 等价于 [log(exp(βR_i) / Σ exp(βR_j))](file:///Users/chaelchael/TreeMatch-RL/flow_grpo-main/scripts/train_sd3_GRPO_Guard.py#181-218)。使用 `log_softmax` 增强了数值稳定性。

---

## 2. 树状采样结构 (3阶 27 分支) ✅

**Draft §4.2**: 三个连续分叉点，每次 SDE 注入 3 组独立噪声 → 1→3→9→27。

**代码** ([tree_sampler.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/sampling/tree_sampler.py#L167-L190)):
```python
for _ in range(self.k):  # k=3
    new_b = branch.clone_for_branch()
    z_new, log_prob, mean, std = flow_sde_step(...)
    ...
    new_branches.append(new_b)
```

**评判**: ✅ **一致**。
- `k=3` 在三个分叉步各产生 3 个分支 → 3³ = 27 条路径
- 前缀共享通过 [clone_for_branch()](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/sampling/tree_sampler.py#34-50) 实现（共享历史，独立未来）
- 验证断言 `assert len(active_branches) == self.k ** len(split_steps)` 保证正确性
- 与 TreeGRPO 的树状采样思路一致，但 TreeGRPO 使用 GRPO 优势估计而非 Softmax-TB

---

## 3. Beta 分布自适应调度 ✅

**Draft §4.2 (Eq. 14)**:
$$t_{split} \sim \mathrm{Beta}(1 + (1-\alpha)\kappa, \; 1 + \alpha\kappa)$$

**代码** ([scheduler.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/sampling/scheduler.py#L85-L111)):
```python
a = 1.0 + (1.0 - alpha) * self.kappa
b = 1.0 + alpha * self.kappa
beta_dist = torch.distributions.Beta(a, b)
fractions = beta_dist.sample((self.num_splits,)).sort().values
```

**评判**: ✅ **完全一致**。
- α 计算 (Eq. 13) 正确实现了 `clip((R̄ - R_min) / (R_max - R_min), 0, 1)`
- κ=0 退化为均匀分布的行为也正确处理
- 额外的 EMA 更新 R_min/R_max 是合理的工程增强

---

## 4. 粒子熵正则化 ⚠️

**Draft §4.3 (Eq. 15)**:
$$\mathcal{L}_{Entropy} = \frac{1}{K(K-1)} \sum_{i \neq j} \exp\left(-\frac{\|\phi(x_i) - \phi(x_j)\|^2}{h}\right)$$

**代码** ([entropy.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/losses/entropy.py#L41-L55)):
```python
dists_sq = torch.cdist(features, features, p=2.0).pow(2)
rbf = torch.exp(-dists_sq / (self.bandwidth + 1e-8))
mask = ~torch.eye(K, dtype=torch.bool, device=features.device)
loss = rbf[mask].mean()  # ← 用 mean() 而非 sum()/K(K-1)
```

> [!WARNING]
> **偏差**: 代码用 `.mean()` 对 mask 后的元素求平均。mask 后共有 `K² - K = K(K-1)` 个元素，所以 `mean()` = `sum() / K(K-1)`，**结果等价**。初看有差异，实际**数学上一致**。
> 
> 但 Draft 中特征映射 φ(x_i) 未明确定义——代码中使用**全局平均池化** (`latents.mean(dim=(-2,-1))`) 将 (C,H,W) → (C,) 作为特征。这是一个合理但简化的选择，Draft 中对 φ 无具体约束。

**评判**: ✅ 核心公式一致（mean 等价于 1/K(K-1)·sum），φ 映射合理。

---

## 5. 参考约束 L_Ref ⚠️

**Draft §4.3 (Eq. 16)**:
$$\mathcal{L}_{Ref} = \sum_{i=1}^K \left\| \frac{1}{T} \log\pi_\theta(\tau_i) - \frac{1}{T} \log\pi_{ref}(\tau_i) \right\|^2$$

**代码** ([reference.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/losses/reference.py#L34-L42)):
```python
norm_current = current_log_probs / max(num_sde_steps, 1)
norm_ref = ref_log_probs / max(num_sde_steps, 1)
loss = ((norm_current - norm_ref) ** 2).sum()
```

**评判**: ✅ **一致**。Draft 公式中 T 未显式定义为 `num_inference_steps`。由于 `log π_θ(τ_i)` 仅在 SDE 步累积非零值（ODE 步 log_prob = 0），代码用 `num_sde_steps`(3) 做归一化是合理的——它除以的是实际有贡献的步数，与 Draft 公式意图一致。

---

## 6. RatioNorm IS ✅ 已修复

**Draft §4.3 (Eq. 18-20)**:

逐步标准化:
$$\log\hat{w}_{i,t} = \sigma_t\sqrt{\Delta t}\left(\log w_{i,t} + \frac{\|\Delta\mu_\theta\|^2}{2\sigma_t^2\Delta t}\right)$$

轨迹级聚合:
$$\log\hat{w}_i = \frac{1}{T}\sum_{t=1}^T \log\hat{w}_{i,t}$$

---

### flow_grpo 参考实现

[train_sd3_GRPO_Guard.py:L915-932](file:///Users/chaelchael/TreeMatch-RL/flow_grpo-main/scripts/train_sd3_GRPO_Guard.py#L915-L932)：

```python
sigma_t = std_dev_t.mean()
ratio_mean_bias = (prev_sample_mean - sample["prev_sample_mean"][:, j]).pow(2) \
                  .mean(dim=tuple(range(1, log_prob.ndim)))      # ← .mean() 因为 log_prob 也用 .mean()
ratio_mean_bias = ratio_mean_bias / (2 * (sqrt_dt.mean() * sigma_t) ** 2)
ratio = torch.exp((log_prob - sample["log_probs"][:, j] + ratio_mean_bias) * (sqrt_dt.mean() * sigma_t))
# ...
policy_loss = policy_loss / (sqrt_dt.mean()**2)
```

---

### 修复前 (旧代码)

```python
def compute_weights(self, current_log_probs, old_log_probs):
    log_ratio = current_log_probs - old_log_probs         # (K,) 轨迹级
    log_ratio_normalized = log_ratio - log_ratio.mean()    # 仅做零均值化
    weights = torch.exp(log_ratio_normalized)
    weights = torch.clamp(weights, 1-ε, 1+ε)
    return weights.detach()
```

缺失 ①偏置修正 ②σ_t·√Δt 缩放 ③逐步标准化 ④loss 归一化。

---

### 修复后 (当前代码)

**5 个文件修改**:

| 文件 | 修改 |
|:----|:----|
| [sde_step.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/sampling/sde_step.py#L101-L152) | [recompute_log_prob](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/sampling/sde_step.py#101-152) 新增返回 `mean`, `std_dev_t`, `sqrt_dt` |
| [ratio_norm.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/losses/ratio_norm.py) | 重写为逐步 RatioNorm |
| [tree_sampler.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/sampling/tree_sampler.py) | Branch 存 `step_means`; [recompute](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/sampling/sde_step.py#101-152) 返回逐步 dict |
| [total_loss.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/losses/total_loss.py) | 接受逐步数据 + `weighted_tb /= sqrt_dt_sq_mean` |
| [train.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/train.py#L193-L225) | 传递 6 项逐步数据到 `loss_fn` |

核心修复代码 ([ratio_norm.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/losses/ratio_norm.py#L72-L96)):

```python
for t in range(T):
    noise_product = sigma_t * sqrt_dt                       # σ_t · √Δt

    log_w = current_step_log_probs[t] - old_step_log_probs[t]   # ① 逐步 log ratio

    delta_mu = current_step_means[t] - old_step_means[t]
    bias = delta_mu.pow(2).sum(dim=tuple(range(1, delta_mu.ndim)))  # ② 偏置 (用 .sum())
    bias = bias / (2.0 * noise_product ** 2)

    log_w_normalized = (log_w + bias) * noise_product           # ③ σ_t·√Δt 缩放

log_w_traj = stack(normalized_ratios).mean(dim=1)               # ④ 轨迹级均值聚合
# total_loss.py: weighted_tb /= sqrt_dt_sq_mean                 # ⑤ loss 归一化
```

> [!IMPORTANT]
> **关键细节**: treematch-rl 的 [log_prob](file:///Users/chaelchael/TreeMatch-RL/flow_grpo-main/scripts/train_sd3_GRPO_Guard.py#181-218) 在空间维度使用 `.sum()` (而 flow_grpo 用 `.mean()`)，因此 bias 也必须使用 `.sum()` 保持一致。flow_grpo 两者都用 `.mean()`，treematch-rl 两者都用 `.sum()`——各自内部一致。

---

### 修复验证

| 步骤 | Draft / flow_grpo | treematch-rl 修复后 |
|:-----|:-----------------|:-------------------|
| ① 偏置修正 `+\|\|Δμ\|\|²/(2σ²Δt)` | ✅ `.mean()` | ✅ `.sum()` (与 log_prob 匹配) |
| ② σ_t·√Δt 缩放 | ✅ | ✅ |
| ③ 逐步独立标准化 | ✅ | ✅ |
| ④ 轨迹级聚合 | N/A (GRPO 无轨迹聚合) | ✅ `.mean(dim=1)` |
| ⑤ loss 归一化 `/ sqrt_dt²` | ✅ | ✅ |

---

## 7. 总损失函数 ✅

**Draft §4.3 (Eq. 21)**:
$$\mathcal{L}_{total} = \frac{1}{K}\sum_{i=1}^K \text{clip}(\hat{w}_i, 1{-}\varepsilon, 1{+}\varepsilon) \cdot \mathcal{L}_{Soft\text{-}TB}^{(i)} + \lambda_1 \mathcal{L}_{Entropy} + \lambda_2 \mathcal{L}_{Ref}$$

**代码** ([total_loss.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/losses/total_loss.py#L60-L80)):
```python
per_path_tb = self.soft_tb.forward_per_path(current_log_probs, rewards)
weights = self.is_module.compute_weights(current_log_probs, old_log_probs)
weighted_tb = (weights * per_path_tb).mean()
total_loss = weighted_tb + self.lambda_entropy * loss_entropy + self.lambda_ref * loss_ref
```

**评判**: ✅ **结构一致**。[(weights * per_path_tb).mean()](file:///Users/chaelchael/TreeMatch-RL/TreeGRPO-main/train.py#479-504) = [(1/K) Σ clip(ŵ_i) · L^(i)](file:///Users/chaelchael/TreeMatch-RL/TreeGRPO-main/train.py#479-504)。

---

## 8. DPM-Solver++ ✅

**Draft §4.4 (Eq. 22-24)**:
- 翻译层: `x_θ = x_t - v_θ · t`
- log-SNR: `λ = ln((1-t)/t)`
- 二阶修正: `D_i = (1 + h/(2h_{i-1})) x₀^(i) - h/(2h_{i-1}) x₀^(i-1)`
- 状态转移: `x_{t_next} = (t_next/t) x_t - (1-t_next)(exp(h)-1) D_i`

**代码** ([dpm_solver.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/sampling/dpm_solver.py#L38-L114)):
```python
def velocity_to_x0(v_pred, x_t, sigma):
    return x_t - v_pred * sigma  # Eq. 22 ✅

def sigma_to_lambda(sigma):
    return math.log((1.0 - sigma) / sigma)  # log-SNR ✅

# 二阶修正 (midpoint)
r = h / (h_prev + 1e-8)
D = (1.0 + r / 2.0) * x0_pred - (r / 2.0) * x0_prev  # Eq. 23 ✅

# 状态转移
x_next = (sigma_next / sigma) * x_t - (1.0 - sigma_next) * (exp(h) - 1.0) * D  # Eq. 24 ✅
```

**评判**: ✅ **完全一致**。翻译层、log-SNR、二阶修正系数、指数积分公式完全匹配。MixGRPO Flash 压缩逻辑也得到了正确实现。

---

## 9. SDE 步进 & log_prob ✅

**Draft (Eq. 6)**:
$$\log\pi_\theta(x_{t-\Delta t}|x_t) = -\frac{\|x_{t-\Delta t} - \mu_\theta\|^2}{2\sigma_t^2\Delta t} - \frac{d}{2}\log(2\pi\sigma_t^2\Delta t)$$

**代码** ([sde_step.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/sampling/sde_step.py#L85-L97)):
```python
log_prob = (
    -((prev_sample - mean) ** 2).sum() / (2.0 * noise_scale ** 2)
    - d * math.log(noise_scale)
    - 0.5 * d * math.log(2.0 * math.pi)
)
```

**评判**: ✅ **一致**。与 `flow_grpo` 的 [sd3_sde_with_logprob.py](file:///Users/chaelchael/TreeMatch-RL/flow_grpo-main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py) 参考实现一致。SDE 漂移修正系数 (`drift_coeff_x`, `drift_coeff_v`) 和噪声项的计算方式也正确。

> [!NOTE]
> 与 flow_grpo 参考实现的一个差异：flow_grpo 对 log_prob 做了 `.mean(dim=tuple(range(1, log_prob.ndim)))`（即在空间维度取均值），而 treematch-rl 使用 `.sum()`（在所有维度求和）。这不影响优化方向，但影响数值量级，需确保 loss 系数与此匹配。

---

## 10. 训练循环逻辑 ⚠️

**代码** ([train.py](file:///Users/chaelchael/TreeMatch-RL/treematch-rl/treematch/train.py#L175-L235)):

> [!WARNING]
> **ref_log_probs 始终等于初始采样时的 log_prob**:
> ```python
> old_log_probs = torch.tensor([b["log_prob_sum"] for b in branches], ...)
> ref_log_probs = old_log_probs.clone().detach()  # ← 从未更新!
> ```
> 在 IS 多次更新中，`old_log_probs` 会更新为前一轮的 `current_log_probs`，但 `ref_log_probs` 始终不变。这意味着 L_Ref 始终约束当前策略相对于**采样时的策略**（而非初始预训练模型）。
> 
> 如果 Draft 的意图是约束相对于预训练参考模型的偏移，则需要单独用参考模型做一次 forward pass 来计算 `ref_log_probs`。

---

## 与参考代码库的对比

### vs flow_grpo
- SDE 步进公式来源一致（均源自 flow_grpo 的 [sd3_sde_with_logprob.py](file:///Users/chaelchael/TreeMatch-RL/flow_grpo-main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py)）
- flow_grpo 使用标准 GRPO 优势估计，treematch-rl 使用 Softmax-TB (正确区分)

### vs TreeGRPO
- 树状采样结构借鉴自 TreeGRPO（前缀共享、分叉机制）
- TreeGRPO 使用 PPO clip 做 reward-maximization，treematch-rl 用 Softmax-TB 做 distribution matching（正确区分）
- TreeGRPO 使用树优势递归回传 ([update_advantages](file:///Users/chaelchael/TreeMatch-RL/TreeGRPO-main/train.py#126-139))，treematch-rl 不做此操作（因为 Softmax-TB 不需要优势估计）

### vs FlowRL
- FlowRL 使用 MLP 拟合配分函数 Z，treematch-rl 通过 Softmax 归一化消除 Z（正确区分，符合 Draft 创新点）

### vs MixGRPO
- DPM Flash 压缩策略借鉴自 MixGRPO
- 自适应调度机制是 TreeMatch-RL 的独立创新（MixGRPO 用固定滑动窗口）

---

## 优先修复建议

| 优先级 | 问题 | 建议 |
|:-------|:----|:-----|
| 🟡 中 | ref_log_probs 逻辑 | 如需真正的预训练参考约束，需单独 forward pass |
| 🟢 低 | log_prob 空间维度 sum vs mean | 确保 loss 系数已针对 sum 模式调优 |
