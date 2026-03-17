# Toy Experiment README

## 实验简介

这个目录下的 toy experiment 用来做一个非常干净的先验验证：

> 不依赖树状采样、ref、entropy、RatioNorm 等训练 trick，只验证 `Softmax-TB` 这件事本身是否能让“模型概率分布”去匹配“奖励诱导分布”，从而避免 mode collapse。

这里我们把：

- 一个二维点 `x in R^2`
- 看成一条“路径”

然后训练一个小 MLP 输出该点的 score：

```python
f_theta(x) -> score(x)
```

再把全网格上的 score 做 softmax，得到模型分布：

```python
p_theta(x_i) = softmax(score(x_i))
```

接着比较两种目标：

1. `Softmax-TB`
   - 让 `p_theta` 去匹配 `exp(beta * reward)` 诱导出的分布
2. `Reward Maximization`
   - 直接最大化 `E_{x ~ p_theta}[R(x)]`

最后看：

- 学到的分布是不是多峰
- 是否覆盖多个波峰
- 与真实目标分布的 `KL / JS` 距离谁更小

如果一切正常，你应该会看到：

- `Softmax-TB` 更接近真实多峰分布
- `Reward-Max` 更容易把概率质量压向最高峰


## 目录结构

```text
toy-experiment/
├── README.md
├── common.py
├── run_softmax_tb.py
├── run_reward_max.py
└── run_compare.py
```

各文件作用：

- [common.py](/Users/lijiaming/Desktop/TreeMatchRL/TreeMatch-RL/toy-experiment/common.py)
  - 公共逻辑
  - 高斯混合 reward
  - MLP 模型
  - 训练函数
  - 指标计算
  - 可视化
- [run_softmax_tb.py](/Users/lijiaming/Desktop/TreeMatchRL/TreeMatch-RL/toy-experiment/run_softmax_tb.py)
  - 只跑 `Softmax-TB`
- [run_reward_max.py](/Users/lijiaming/Desktop/TreeMatchRL/TreeMatch-RL/toy-experiment/run_reward_max.py)
  - 只跑 `Reward-Max` baseline
- [run_compare.py](/Users/lijiaming/Desktop/TreeMatchRL/TreeMatch-RL/toy-experiment/run_compare.py)
  - 两种方法都跑
  - 自动生成对比图和 summary


## 实验框架

### 1. 目标分布

我们手工定义了一个二维 Gaussian mixture，包含 5 个波峰，高度和宽度不同：

- A: `(2.5, 2.5)`
- B: `(-2.5, 2.5)`
- C: `(-2.5, -2.5)`
- D: `(2.5, -2.5)`
- E: `(0.0, 0.0)`

真实目标分布记为：

```python
p_target(x)
```

reward 直接定义成：

```python
R(x) = log p_target(x)
```

这样做的好处是：

- 真实分布已知
- `KL / JS` 好算
- 实验解释非常直接

### 2. 模型

模型是一个很小的 MLP：

```python
R^2 -> R
```

输入是二维坐标，输出是该点的 score。

### 3. 两种训练目标

#### Softmax-TB

```python
log_p_theta = log_softmax(score)
log_p_target = log_softmax(beta * reward)
L_tb = mean((log_p_theta - log_p_target)^2)
```

这个目标想做的事是：

> 让模型分布去匹配奖励诱导分布，而不是只追最高 reward 的点。

#### Reward-Max baseline

```python
L_max = -sum p_theta(x_i) * reward_i
```

这个目标想做的事是：

> 直接最大化平均奖励。

它更容易 collapse 到最高峰附近。


## 环境配置

## 1. 推荐环境

建议用 Python 3.10 或 3.11，Mac 上可以直接用虚拟环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

## 2. 需要的依赖

最少需要：

```bash
pip install torch matplotlib
```

如果你想在 Apple Silicon 上优先用 MPS，通常直接安装官方 `torch` 就可以；如果 MPS 不可用，脚本会自动回退到 CPU。

可选：

```bash
pip install jupyter
```

如果你想后面在 notebook 里画图分析，可以再装这个。

## 3. 快速检查

装完后可以先试：

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import matplotlib; print(matplotlib.__version__)"
```


## 如何运行

## 1. 只跑 Softmax-TB

```bash
python3 toy-experiment/run_softmax_tb.py --device auto
```

## 2. 只跑 Reward-Max baseline

```bash
python3 toy-experiment/run_reward_max.py --device auto
```

## 3. 跑完整对比实验

```bash
python3 toy-experiment/run_compare.py --device auto
```

如果你明确要用 Mac 的 MPS：

```bash
python3 toy-experiment/run_compare.py --device mps
```

如果你想先做一个很快的 smoke test：

```bash
python3 toy-experiment/run_compare.py \
  --device cpu \
  --train-steps 50 \
  --grid-size 61 \
  --eval-samples 1000 \
  --run-name smoke_test
```


## 输出内容

每次运行都会在：

```text
toy-experiment/outputs/<run_name>_<timestamp>/
```

下面生成结果。

常见输出包括：

- `*_metrics.json`
  - 单个方法的训练曲线和最终指标
- `comparison_summary.json`
  - 两个方法的最终指标汇总
- `softmax_tb_overview.png`
  - Softmax-TB 的单独可视化
- `reward_max_overview.png`
  - Reward-Max 的单独可视化
- `comparison_overview.png`
  - 两种方法的总对比图


## 图怎么看

可视化里主要看这几块：

## 1. Reward Landscape

目标的多峰 reward 地形图。  
这个图告诉你真实应该学成什么样。

## 2. Learned Distribution

模型最终学到的概率分布热图。

重点看：

- 是否有多个峰
- 是否只剩一个最强峰
- 峰的位置是否和目标对齐

## 3. Sample Rollout

从最终学到的分布里采样很多点后画出来的散点图。

这个图最直观：

- `Softmax-TB` 如果对，会在多个峰附近都有样本
- `Reward-Max` 更容易堆到高峰

## 4. Mode Proportion Comparison

每个峰的目标质量 vs 学到的质量对比。

这张图最适合做论文式结论：

- 是否只学到一个峰
- 是否比例接近真实权重

## 5. Training Curves / Distribution Distance

重点看：

- `KL`
- `JS`
- `avg_reward`
- `peak_prob`

通常：

- `Softmax-TB` 的 `KL / JS` 应该更低
- `Reward-Max` 的 `avg_reward` 可能不差，但 `peak_prob` 更容易变高，说明更集中、更 collapse


## 调参建议

## 1. 最重要的参数

### `--beta`

只作用于 `Softmax-TB`。

作用：

- `beta` 越小，目标分布越平
- `beta` 越大，目标分布越尖，越偏向高 reward 峰

建议：

- 第一版先用 `1.0`
- 然后做一个小消融：`0.5 / 1.0 / 2.0 / 5.0`

### `--grid-size`

决定二维空间离散得多细。

作用：

- 越大越精细
- 但也越慢

建议：

- 快速测试：`61`
- 正式图：`121`
- 更平滑但更慢：`151`

### `--train-steps`

训练总步数。

建议：

- smoke test：`50 ~ 100`
- 正式第一版：`1000`
- 如果图还不够稳定，可以试 `2000`

### `--hidden-dim`

MLP 容量。

建议：

- 默认 `64`
- 如果你觉得峰学不出来，可以试 `128`
- 不建议一开始就很大，没有必要

### `--lr`

学习率。

建议：

- 默认 `1e-3`
- 如果训练震荡太大，降到 `5e-4`
- 如果收敛太慢，可以试 `2e-3`

### `--eval-samples`

最终散点图的采样数量。

建议：

- 快速看图：`2000`
- 正式展示：`10000`


## 2. 推荐的实验顺序

### 第一轮：确认代码能跑

```bash
python3 toy-experiment/run_compare.py \
  --device auto \
  --train-steps 100 \
  --grid-size 61 \
  --eval-samples 1000 \
  --run-name quick_check
```

### 第二轮：出第一张正式图

```bash
python3 toy-experiment/run_compare.py \
  --device auto \
  --train-steps 1000 \
  --grid-size 121 \
  --eval-samples 10000 \
  --beta 1.0 \
  --run-name main_result
```

### 第三轮：做 beta 消融

```bash
python3 toy-experiment/run_softmax_tb.py --beta 0.5 --run-name beta_05
python3 toy-experiment/run_softmax_tb.py --beta 1.0 --run-name beta_10
python3 toy-experiment/run_softmax_tb.py --beta 2.0 --run-name beta_20
python3 toy-experiment/run_softmax_tb.py --beta 5.0 --run-name beta_50
```


## 如何判断实验是不是成功

如果实验成功，你应该看到：

### Softmax-TB

- `KL(p_theta || p_target)` 更低
- `JS` 更低
- `mode_coverage` 更高
- 多个峰都有明显概率质量
- `sample rollout` 散点图是多峰的

### Reward-Max

- 平均奖励不一定低
- 但更容易只集中在最高峰或少数峰
- `KL / JS` 更大
- `peak_prob` 更高


## 常见问题

## 1. 报错 `No module named torch`

说明还没装 PyTorch：

```bash
pip install torch
```

## 2. 报错 `No module named matplotlib`

说明还没装画图库：

```bash
pip install matplotlib
```

## 3. `--device mps` 跑不起来

说明当前环境不支持 MPS，直接改成：

```bash
--device cpu
```

或者用：

```bash
--device auto
```

## 4. 图上还是只有一个峰

可以优先试：

1. 降低 `beta`
2. 增加 `train_steps`
3. 增大 `hidden_dim`
4. 检查 `comparison_overview.png` 里 Reward Landscape 是否正常

## 5. 训练很慢

优先调小：

- `grid-size`
- `eval-samples`
- `train-steps`


## 和论文/大实验的关系

这个 toy 的定位非常明确：

- 它不是完整 TreeMatch-RL 的复刻
- 它不是大模型训练替代品
- 它是一个**先验验证实验**

它要回答的问题是：

> “Softmax-TB 让路径概率匹配奖励分布”这个核心思想，能不能天然得到多峰、多样化的解？

如果这个 toy 成立，你后面再加 tree sampling、ref、entropy、RatioNorm，就可以自然解释成：

- 核心思想已经成立
- 这些额外机制只是让大模型训练更可行、更稳定


## 一句话建议

先跑这个：

```bash
python3 toy-experiment/run_compare.py --device auto --run-name first_real_run
```

然后重点看：

- `comparison_overview.png`
- `comparison_summary.json`

这两个文件最能快速告诉你：

**你的 Softmax-TB 核心 idea 在这个 toy 上到底有没有把多峰分布学出来。**
