# Softmax-TB Core Toy Experiment（先验验证版，可直接在 Mac 上跑）

## 1. 实验目的

这个 toy experiment 的目标，不是复刻完整 `treematch-rl` 训练流程，而是**只验证你最核心的观点**：

> 用 `Softmax-TB` 让“路径概率分布”去匹配“奖励诱导分布”，本身就能避免只追最高峰，从而得到多峰、多样化的解。

这里我建议把论证切得非常干净：

- **核心 idea**：`路径概率匹配奖励分布`
- **不放进第一版的内容**：树状采样、ref、entropy、RatioNorm

原因很简单：

- 树状采样是训练 trick，用来提高大模型训练效率
- ref / entropy 是正则和工程稳定项
- 你这次 toy 的目的，是证明 **Softmax-TB 这个概率匹配思想本身成立**

所以第一版 toy 最好是：

> 二维点 = 一条路径  
> MLP 输出这个点的分数 = 路径 logit / 路径 log-prob 的代理  
> 用 Softmax-TB 训练这个分数场  
> 训练后从学到的分布里采样很多点，看是不是自然形成多峰覆盖


## 2. 这个 toy 和当前 TreeMatch-RL 的对应关系

为了让 toy 和你现在的代码理念保持一致，可以做下面这个一一对应：

| TreeMatch-RL 概念 | Toy 中的对应物 |
|---|---|
| 一条完整 diffusion path | 一个二维点 `x in R^2` |
| path log_prob | MLP 对该点输出的标量 score |
| reward model 打分 | 手工构造的二维 reward landscape |
| Softmax-TB 在组内匹配 `log_p` 与 `beta * reward` | 在二维候选点集合上匹配模型分布与奖励分布 |
| rollout 很多 branch 看覆盖情况 | 从训练好的分布里采样很多二维点看是否多峰 |

也就是说，这个 toy 不是在验证“扩散采样工程有没有做好”，而是在验证：

> **如果模型学的是一整个候选集合上的概率分布，而训练目标是让这个概率分布匹配奖励分布，那么它会不会天然比 reward maximization 更不容易 collapse。**


## 3. 最推荐的实验版本：离散网格上的连续评分模型

为了保证 Mac 上最容易跑通，我建议第一版不要做复杂 rollout sampler，而是做一个**离散 2D 网格上的概率模型**。

### 3.1 空间

定义二维区域：

```python
x, y in [-4, 4]
```

把它离散成一个网格，比如：

```python
grid_size = 121
```

则总点数大约是：

```python
121 * 121 = 14641
```

这对 Mac 来说非常轻。

### 3.2 模型

训练一个小 MLP：

```python
f_theta: R^2 -> R
```

输入一个二维点 `x`，输出一个标量 `score_theta(x)`。

这个分数不必强行解释成“严格的连续 log-density”，在 toy 里把它当成：

> **该点对应的路径 logit / 路径概率的未归一化分数**

就够了。

然后在整张网格上做 softmax：

```python
p_theta(x_i) = softmax(score_theta(x_i))
```

于是你就得到了一个定义在整个 2D 平面离散网格上的概率分布。

这一步特别重要，因为它让整个实验：

- 不需要 diffusion
- 不需要 MCMC
- 不需要 rollout ODE/SDE
- 不需要 proposal correction

但仍然保留了你最想证明的那件事：

**模型学一个分布，而不是只学一个最优点。**


## 4. 奖励函数设计

## 4.1 多峰 reward landscape

定义 5 个二维高斯峰，每个峰高度不同：

| Mode | Center | Weight | Sigma |
|---|---|---|---|
| A | `(2.5, 2.5)` | `0.30` | `0.45` |
| B | `(-2.5, 2.5)` | `0.25` | `0.45` |
| C | `(-2.5, -2.5)` | `0.20` | `0.45` |
| D | `(2.5, -2.5)` | `0.15` | `0.45` |
| E | `(0.0, 0.0)` | `0.10` | `0.70` |

定义真实目标分布：

```python
p_target(x) = sum_m w_m * N(x | mu_m, sigma_m^2 I)
```

### 4.2 最干净的 reward 定义

为了让“真实分布”清晰、KL 也好算，我强烈建议把 reward 直接定义成：

```python
R(x) = log p_target(x)
```

这样有一个非常大的好处：

如果你把 `beta = 1`，那么 Softmax-TB 的目标就变成：

```python
p_theta(x)  match  p_target(x)
```

也就是说，这个 toy 里你不是“间接地猜测” Softmax-TB 会不会多峰，而是：

> 直接测试它能不能把学到的分布逼近一个已知的多峰真实分布。

这会让你的实验非常干净，也更适合作为论文前的先验验证。


## 5. 两种训练目标

## 5.1 Method A：你的核心方法，Softmax-TB 分布匹配

在整张网格上：

```python
score_i = f_theta(x_i)
log_p_theta = log_softmax(score_i)
log_p_target = log_softmax(beta * reward_i)
```

损失定义为：

```python
L_tb = mean((log_p_theta - log_p_target)^2)
```

如果：

```python
reward_i = log p_target(x_i)
beta = 1
```

那么这个 loss 的目标就是直接让模型分布去逼近真实分布。

### 这个方法想证明什么

它证明的是：

> 如果我们不是直接最大化奖励，而是让模型分布去匹配奖励诱导出的分布，那么模型会保留多模态结构，而不是全部坍缩到单一最高峰。


## 5.2 Method B：奖励最大化 baseline

对照组就用最简单、最像“mode collapse 来源”的目标：

```python
p_theta = softmax(score_i)
L_max = -sum_i p_theta(x_i) * reward_i
```

也就是最大化：

```python
E_{x ~ p_theta}[R(x)]
```

这个目标的最优倾向非常清楚：

- 它不关心分布是否按比例覆盖多峰
- 它只关心平均 reward 尽量高
- 所以它天然更倾向于把质量压到 reward 最高的峰附近

这正好对应你想对比的“奖励最大化会更容易 collapse”。


## 6. 为什么这个设计比之前版本更好

相比之前那版 tree/diffusion toy，这一版更适合作为第一性验证，原因是：

### 6.1 更贴近你真正想证明的命题

你要证明的不是：

- 树状采样是不是一定更好
- ref / entropy 有没有帮助
- diffusion toy 能不能训起来

而是：

> **路径分布匹配奖励分布，这个思想本身是否能导向多解、多峰覆盖。**

这版实验正好只打这个靶心。

### 6.2 在 Mac 上极容易实现

这一版只需要：

- `torch`
- `numpy`
- `matplotlib`

不需要：

- 多卡
- MPS 特殊优化
- 复杂 rollout
- ODE/SDE log-prob
- reward model

### 6.3 结果特别容易解释

因为真实目标分布就是你自己定义的 Gaussian mixture，所以：

- 看图就知道有没有 collapse
- 算 KL / JS 也容易
- 不会陷入“大模型训练里到底是谁在起作用”的混杂解释


## 7. 具体可执行实验流程

## 7.1 构建固定网格

```python
xs = torch.linspace(-4, 4, 121)
ys = torch.linspace(-4, 4, 121)
grid = {(x_i, y_j)}
```

得到全部候选点：

```python
X in R^{N x 2}, N = 14641
```

## 7.2 计算真实 reward / 真实分布

对每个点算：

```python
reward_i = log p_target(x_i)
```

然后归一化得到真实离散分布：

```python
p_target_grid = softmax(reward_i)   # beta = 1 时
```

## 7.3 训练 Softmax-TB 模型

每一步：

1. MLP 输出全网格 `score_i`
2. 计算 `log_p_theta = log_softmax(score_i)`
3. 计算 `log_p_target = log_softmax(beta * reward_i)`
4. 最小化 `mean((log_p_theta - log_p_target)^2)`

## 7.4 训练 reward-max baseline

每一步：

1. MLP 输出全网格 `score_i`
2. `p_theta = softmax(score_i)`
3. 最大化 `sum p_theta * reward`

## 7.5 训练后 rollout

训练完成后，从学到的离散分布里采样很多点：

```python
x ~ Categorical(p_theta)
```

比如采：

```python
num_samples = 10000
```

然后把采样点画成散点图或 2D heatmap。

这里的“rollout”不再是 diffusion rollout，而是：

> 从训练好的路径概率分布里抽样很多二维点，检查它是不是多峰。


## 8. 评估指标

### 8.1 最核心指标

1. **KL 散度**

```python
KL(p_theta || p_target)
```

这个是你最想要的主指标，因为它直接回答：

> 学到的分布是不是接近真实多峰分布。

2. **JS 散度**

用于更稳定地看分布距离。

3. **Mode coverage**

把每个采样点分配到最近的峰，统计：

- 覆盖到多少个峰
- 每个峰的样本比例是多少

4. **平均奖励**

这个可以保留，但它不是最关键指标。

### 8.2 预期现象

### Softmax-TB

- KL 更低
- JS 更低
- 覆盖多个峰
- 峰的样本比例更接近真实权重

### Reward maximization

- 平均奖励可能不低
- 但会明显偏向最高峰
- KL 更大
- 多样性更差


## 9. 第一版推荐超参

```python
config = {
    "seed": 42,
    "device": "mps_or_cpu",
    "grid_size": 121,
    "hidden_dim": 64,
    "lr": 1e-3,
    "train_steps": 1000,
    "beta": 1.0,
    "eval_samples": 10000,
}
```

### 为什么这么设

1. `grid_size=121`
   - 足够平滑
   - 对 Mac 依然很轻

2. `hidden_dim=64`
   - 足够学 5 个峰
   - 不会太重

3. `lr=1e-3`
   - 对小 MLP 一般很稳

4. `train_steps=1000`
   - 通常足够看到明显分布差异

5. `beta=1.0`
   - 让 Softmax-TB 的目标直接对应真实混合分布


## 10. Mac 落地建议

建议直接做成一个单文件：

```text
experiments/toy_softmax_tb/run_toy.py
```

里面包含：

- Gaussian mixture target
- reward 计算
- MLP
- Softmax-TB trainer
- reward-max trainer
- evaluation
- plotting

运行方式：

```bash
python experiments/toy_softmax_tb/run_toy.py --method softmax_tb --device mps
python experiments/toy_softmax_tb/run_toy.py --method reward_max --device mps
python experiments/toy_softmax_tb/run_toy.py --method both --device mps
```

设备逻辑：

```python
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```


## 11. 这版实验写进论文时怎么表述

你可以把这个 toy 的论点写得非常清楚：

> 为了隔离验证 TreeMatch-RL 的核心思想，我们设计了一个二维合成实验。在该实验中，一个二维点被视作一条“路径”，模型学习该点的未归一化路径概率分数。我们比较了两种目标：  
> 1. 用 Softmax-TB 让模型概率分布匹配奖励诱导分布；  
> 2. 直接最大化期望奖励。  
> 结果表明，前者可以恢复多峰目标分布，而后者更易坍缩到少数高奖励峰值。  

这样你就把：

- Softmax-TB 的核心思想
- 多样性不是靠 trick 硬撑
- reward maximization 易 collapse

这三件事讲清楚了。


## 12. 第二阶段再加什么

如果第一版跑通，第二阶段再加：

1. `beta in {0.5, 1.0, 2.0, 5.0}` 消融
2. 峰数从 `5 -> 8 -> 12`
3. 峰高差异更极端的场景
4. 再做一个“tree candidate set vs flat candidate set”的训练 trick 对比

注意这里的顺序很重要：

- **第一阶段**：证明 `Softmax-TB` 核心 idea 成立
- **第二阶段**：再说明 tree sampling 等 trick 如何帮助训练


## 13. 最终建议

如果你这周就想在 Mac 上把 toy 跑出来，我建议你就按这个版本做：

- 二维 Gaussian mixture
- reward = `log p_target`
- MLP 直接输出点分数
- Softmax-TB vs reward maximization
- 用 KL / JS / mode coverage 做结论

这版是我认为：

> **最贴合你当前论文核心观点、同时实现成本最低、最不容易被工程细节干扰的 toy experiment。**
