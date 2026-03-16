# Flow-GRPO / MixGRPO / TreeGRPO 评测机制详解

## 1. 先说结论

如果只回答一句：

- `flow_grpo` / `GRPO-Guard` 是这三个里评测链路最完整的。它有明确的 `test` 集、周期性 `eval()`、并且在 `GenEval` 任务下会同时记录更像 benchmark 的 `accuracy / strict_accuracy`。
- `MixGRPO` 有“正式离线评测脚本”，但核心还是把生成结果再喂给一组奖励模型打分。它更像一个 `reward-model suite`，不是单一的公共 benchmark。
- `TreeGRPO` 当前这个仓库版本没有独立测试集评测脚本，主要是在训练过程中记录 `HPSv2` 的 `reward/mean`、`reward/std` 和 PPO/GRPO 训练诊断指标。

所以，如果你问“他们是如何做 eval 的”：

- `flow_grpo`：训练中周期性在 `test` split 上跑生成，再算 reward / benchmark 指标。
- `MixGRPO`：先离线生成测试集图片，再跑单独的 `eval_reward.py` 统一打分。
- `TreeGRPO`：当前仓库基本没有单独 eval，更多是训练监控。

如果你再问“这些指标都是行业公认的吗”：

- `GenEval accuracy / strict_accuracy`：更接近“公认 benchmark”。
- `HPSv2 / PickScore / ImageReward / CLIPScore`：是文生图里非常常见的自动评测或奖励模型，属于“公认 proxy 指标”，常用，但不是金标准。
- `OCR reward`：是文本渲染任务很合理的任务型指标，但不是通用 benchmark。
- `UnifiedReward`：更像较新的 VLM-as-a-judge 指标，有用，但没有前面几类那么“老牌”。
- `loss / kl / clipfrac / ratio_mean`：这类只是训练诊断，不是生成质量 benchmark。

## 2. 我是按哪些代码判断的

我主要对照了这些文件：

- `flow_grpo-main/scripts/train_sd3.py`
- `flow_grpo-main/scripts/train_sd3_GRPO_Guard.py`
- `flow_grpo-main/config/grpo.py`
- `flow_grpo-main/config/grpo_guard.py`
- `flow_grpo-main/flow_grpo/rewards.py`
- `flow_grpo-main/flow_grpo/ocr.py`
- `flow_grpo-main/flow_grpo/pickscore_scorer.py`
- `flow_grpo-main/flow_grpo/clip_scorer.py`
- `MixGRPO-main/fastvideo/train_grpo_flux.py`
- `MixGRPO-main/fastvideo/eval/eval_reward.py`
- `MixGRPO-main/fastvideo/models/reward_model/utils.py`
- `MixGRPO-main/fastvideo/models/reward_model/pick_score.py`
- `MixGRPO-main/scripts/inference/inference_flux.sh`
- `MixGRPO-main/scripts/evaluate/eval_reward.sh`
- `TreeGRPO-main/train.py`
- `TreeGRPO-main/reward_models/hps.py`
- `TreeGRPO-main/configs/base.yaml`

下面按项目展开。

---

## 3. Flow-GRPO / GRPO-Guard：怎么做 eval

### 3.1 入口在哪里

代表性实现是：

- `flow_grpo-main/scripts/train_sd3.py`
- `flow_grpo-main/scripts/train_sd3_GRPO_Guard.py`

这两个脚本都有独立的 `eval(...)` 函数。`GRPO-Guard` 的评测流程和 `Flow-GRPO` 基本一致，差别主要在训练更新时的 `RatioNorm / Gradient Reweight`，不是评测脚本本身。

配置文件里会指定：

- 用哪个数据集
- `train/test` batch size
- 每隔多少 epoch 评测一次
- 评测时用什么 reward 函数

例如：

- `config/grpo.py` 里 `general_ocr_sd3()` 用 `dataset/ocr`，reward 是 `ocr`
- `config/grpo.py` 里 `geneval_sd3()` 用 `dataset/geneval`，reward 是 `geneval`
- `config/grpo.py` 里 `pickscore_sd3()` 用 `dataset/pickscore`，reward 是 `pickscore`
- `config/grpo_guard.py` 也是同一套思路，只是超参不同

### 3.2 它的 eval 流程是怎样的

`train_sd3.py` 里这条链路很清楚：

1. 先根据 `config.prompt_fn` 载入 `train_dataset` 和 `test_dataset`。
2. `general_ocr` 任务读 `train.txt / test.txt`。
3. `geneval` 任务读 `train_metadata.jsonl / test_metadata.jsonl`。
4. 训练时每到 `epoch % eval_freq == 0`，就调用一次 `eval(...)`。
5. `eval(...)` 里遍历整个 `test_dataloader`。
6. 对每个 batch 先做文本编码，再生成图片。
7. 然后把生成图片和 prompt 送进 `reward_fn`。
8. 最后把所有 GPU 的结果 gather 后求均值，记到 WandB。

这里有两个很关键的实现细节：

- 评测用的是 `test` split，不是训练 batch 混着看。
- 评测时显式传了 `noise_level=0`，这意味着 eval 是“去掉额外采样噪声”的，更接近确定性 ODE rollout，而不是训练时那种带探索噪声的采样。

也就是说，`flow_grpo` 的 eval 不是“训练顺手看一眼 reward”，而是一个单独的 held-out test 评测回路。

### 3.3 它训练中还会记录什么

除了 test eval，训练阶段也会记录一套 online reward 监控：

- 先在当前 batch 上采样图片
- 再异步算 reward
- 再把 reward 记录到 WandB

这一套是“训练 proxy 监控”，不是正式 test 评测。

它还会额外记录一些 per-prompt 统计：

- `group_size`
- `trained_prompt_num`
- `zero_std_ratio`
- `reward_std_mean`

这些指标主要是为了看同 prompt 的 reward 分布是否塌了、同组样本方差是不是太小，属于训练稳定性分析，不是最终生成质量 benchmark。

### 3.4 Flow-GRPO 里到底有哪些评测指标

`flow_grpo-main/flow_grpo/rewards.py` 里统一由 `multi_score(...)` 把各个 reward 拼起来。这里要注意两个层次：

1. 单个 reward 模块自己的原始分数。
2. `avg` 这个总分。

其中 `avg` 不是严格数学意义上的“平均数”，它其实是：

`avg = sum(weight_i * score_i)`

所以更准确地说，它是“加权总 reward”，只是代码里把它叫成了 `avg`。

#### 3.4.1 OCR

OCR 的公式在 `flow_grpo-main/flow_grpo/ocr.py` 里非常直接：

- 先从 prompt 里抽出引号中的目标文本
- 用 PaddleOCR 识别图片中的文字
- 把识别文本和目标文本都转成小写、去空格
- 算编辑距离 `dist`
- 最后 reward = `1 - dist / len(prompt)`

这意味着：

- 完全识别对，reward 接近 `1`
- 完全不对，reward 接近 `0`

它本质上是一个“归一化编辑距离分数”，很适合文本渲染任务。

#### 3.4.2 PickScore

`flow_grpo-main/flow_grpo/pickscore_scorer.py` 用的是：

- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- `yuvalkirstain/PickScore_v1`

流程是：

- 分别提 image/text embedding
- 做归一化
- 用 `logit_scale * text_emb @ image_emb`
- 最后代码里除以 `26`

所以它在这个仓库里被压到了一个近似 `0~1` 的区间附近，便于做 reward。

这和论文里常见的原始 PickScore 标尺不是完全同一数值范围，所以看结果时要先分清“是不是这个仓库自己的归一化版本”。

#### 3.4.3 CLIPScore

`flow_grpo-main/flow_grpo/clip_scorer.py` 的实现是：

- 用 `openai/clip-vit-large-patch14`
- 取 `outputs.logits_per_image.diagonal()`
- 再除以 `30`

本质上仍然是 image-text 对齐相似度，只是也做了一个简单缩放。

#### 3.4.4 ImageReward

`flow_grpo-main/flow_grpo/imagereward_scorer.py` 直接调用 `ImageReward` 模型的 `inference_rank` 返回 reward。

这是典型的“人类偏好代理模型分数”。

#### 3.4.5 GenEval

这是 `flow_grpo` 里最值得单独讲的。

`geneval_score(...)` 从远端服务拿回了五类东西：

- `scores`
- `rewards`
- `strict_rewards`
- `group_rewards`
- `group_strict_rewards`

然后 `multi_score(...)` 会把它们映射成：

- `geneval`：对应 `scores`
- `accuracy`：对应 `rewards`
- `strict_accuracy`：对应 `strict_rewards`
- 各种 `xxx_accuracy`
- 各种 `xxx_strict_accuracy`

这里最容易混淆：

- `geneval` 这个字段更像“训练优化用的 dense reward”
- `accuracy / strict_accuracy` 才更像真正 benchmark 意义上的评测指标

从字段命名看，我更建议你把它理解成：

- `geneval`：优化信号
- `accuracy / strict_accuracy`：评测输出

尤其 `strict_accuracy`，在组合属性、计数、绑定这些任务上，比单纯的 dense reward 更像论文表格里拿来横向比较的分数。

#### 3.4.6 UnifiedReward

如果启用 `unifiedreward`，代码会：

- 把图像送到一个 VLM judge
- 要求模型输出 `Final Score: 1~5`
- 再把分数除以 `5`

所以最终落到 `0~1`

这类分数本质是“多模态大模型裁判分”，不是传统 benchmark accuracy。

### 3.5 Flow-GRPO 的日志长什么样

在 test eval 阶段，WandB 会记：

- `eval_reward_geneval`
- `eval_reward_accuracy`
- `eval_reward_strict_accuracy`
- `eval_reward_pickscore`
- `eval_reward_ocr`
- `eval_reward_clipscore`
- `eval_reward_avg`
- 以及最后一个 batch 的 `eval_images`

在 train 阶段，WandB 会记：

- `reward_geneval`
- `reward_accuracy`
- `reward_strict_accuracy`
- `reward_pickscore`
- `reward_ocr`
- `reward_avg`

再加上一堆训练相关的统计量。

### 3.6 这些指标算不算“行业公认”

我的判断是：

| 指标 | 地位 |
| --- | --- |
| `GenEval accuracy / strict_accuracy` | 更像公共 benchmark，文生图组合泛化里比较公认 |
| `PickScore` | 很常见的偏好代理指标，业内常用 |
| `ImageReward` | 很常见的偏好代理指标，业内常用 |
| `CLIPScore` | 老牌自动对齐指标，业内常用，但大家也知道它有局限 |
| `OCR reward` | 很合理的任务型指标，但只适合文字渲染任务 |
| `UnifiedReward` | 新一些的 judge-based 指标，有价值，但没前面几种那么稳固 |
| `avg` | 只是仓库内部的加权总 reward，不是公共标准 |

所以更准确的说法不是“全都行业公认”，而是：

- 有些是“公认 benchmark”
- 有些是“公认 proxy”
- 有些是“任务专用”
- 有些只是“训练内部聚合分数”

### 3.7 GRPO-Guard 和 Flow-GRPO 在 eval 上有没有本质区别

基本没有。

`GRPO-Guard` 的 `train_sd3_GRPO_Guard.py` 也会：

- 建 `test_dataset`
- 周期性调用 `eval(...)`
- 在 `eval(...)` 里生成 test 图像
- 调相同的 `reward_fn`
- 记录 `eval_reward_{key}`

所以它和 `Flow-GRPO` 的区别主要在训练更新公式，而不是“换了一套评测体系”。

---

## 4. MixGRPO：怎么做 eval

### 4.1 先区分两件事

`MixGRPO` 代码里有两种“看起来像评测”的东西：

1. 训练时在线算 reward，并记到 WandB / txt。
2. 真正的离线测试脚本 `eval_reward.py`。

这两个一定要分开。

### 4.2 训练时它其实在看什么

`MixGRPO-main/fastvideo/train_grpo_flux.py` 的训练逻辑是：

1. 取 prompt embedding。
2. 用当前模型采样图像。
3. 解码出图像。
4. 立刻调用 `compute_reward(...)` 算奖励。
5. 用这些 reward 归一化成 advantages。
6. 再做 GRPO 更新。

训练中记录到日志里的主要有：

- `train_loss`
- `policy_loss`
- `kl_loss`
- `clip_frac`
- `reward` 或 `reward_{model_name}`

这里的 `reward` 只是当前训练采样出来的 online reward，不是独立测试集结果。

所以如果你只看训练 WandB 曲线，它更像“online optimization monitor”，不是严格意义的 eval benchmark。

### 4.3 MixGRPO 的正式离线评测怎么跑

它的官方流程是两段式：

#### 第一步：先生成测试图片

`scripts/inference/inference_flux.sh` 会调用 `fastvideo/sample/sample_flux.py`：

- 输入 `data/prompts_test.txt`
- 生成图片到输出目录
- 同时写一个 JSON

这个 JSON 长这样：

```json
[
  {
    "image": "xxx/0.jpg",
    "prompt": "..."
  }
]
```

#### 第二步：再打分

`scripts/evaluate/eval_reward.sh` 会调用 `fastvideo/eval/eval_reward.py`：

1. 读取上一步的 JSON。
2. 分布式加载图片。
3. 把每张图和 prompt 送进一个或多个 reward model。
4. 保存逐样本 JSON。
5. 再额外保存一个 `_mean.txt`，写每个 reward model 的均值。

所以 `MixGRPO` 的正式评测不是“训练脚本内部顺便评一下”，而是：

- 先离线生成
- 再离线打分

### 4.4 MixGRPO 里有哪些评测指标

`eval_reward.py` 支持这些 reward model：

- `hpsv2`
- `image_reward`
- `clip_score`
- `pick_score`
- `unified_reward`
- `all`

如果是 `all`，它会把这些模型全跑一遍，然后分别输出每个模型的分数。

也就是说，`MixGRPO` 并没有像 `GenEval strict_accuracy` 这种单一 benchmark 主指标；它更像一套并列的 reward-model report。

#### 4.4.1 HPSv2

`fastvideo/models/reward_model/hps_score.py` 的实现本质上就是：

- image/text 编码
- 取 `image_features @ text_features.T`

这是 HPSv2 的典型用法。

#### 4.4.2 CLIPScore

`fastvideo/models/reward_model/clip_score.py` 用 `open_clip`：

- 编图
- 编文
- 做归一化
- 算相似度

这是典型的自动对齐 proxy。

#### 4.4.3 ImageReward

`fastvideo/models/reward_model/image_reward.py` 直接调用 `ImageReward` 的 reward。

#### 4.4.4 PickScore

这里有一个非常容易忽略的细节：

训练时 `PickScoreRewardModel` 返回的是：

`(raw_score - mean) / std`

也就是 z-score 版本，默认 `mean=18, std=8`。

但在离线评测 `eval_reward.py` 里，它写 JSON 时又把它变回了：

`(z * 8 + 18) / 100`

所以：

- 训练 reward 的尺度
- 离线评测输出的尺度

不是同一个数轴。

如果你直接把训练曲线上的 `reward_pickscore` 和 `_mean.txt` 里的 PickScore 均值放在一起比较，会误判。

#### 4.4.5 UnifiedReward

`fastvideo/models/reward_model/unified_reward.py` 支持两种解析方式：

- `score`：从输出里提 `Final Score`
- `semantic`：从输出里提 `Alignment Score`

而 `scripts/evaluate/eval_reward.sh` 默认给的是：

- `unified_reward_default_question_type="semantic"`

所以常见离线评测其实是在取 `Alignment Score (1-5)`。

这点也很重要，因为：

- `score` 模式更像综合图文一致性 + 质量
- `semantic` 模式更偏语义对齐

你看到的 `UnifiedReward` 分数到底代表什么，取决于这里的配置。

### 4.5 MixGRPO 的多奖励混合，和 eval 不是一回事

训练脚本里有两种多奖励混合策略：

- `reward_aggr`
- `advantage_aggr`

它们的作用是“怎么从多个 reward model 构造训练优势”：

#### `reward_aggr`

先加权合成一个总 reward，再按组标准化成 advantage。

#### `advantage_aggr`

先对每个 reward model 单独做组内标准化，再按权重把多个 advantage 加起来。

这两种会影响训练行为，但不会改变离线 `eval_reward.py` 最终输出“每个模型分数”的方式。

所以不要把“训练里如何混奖励”误以为“测试里如何算最终 benchmark”。

### 4.6 MixGRPO 离线评测还有一个很重要的统计口径

`compute_reward(...)` 有 `success` 机制。

尤其当你同时评多个模型时：

- 只有当所有模型都成功返回结果，`merged_successes[i]` 才会是 `1`
- `_mean.txt` 里的均值只会对这些 `success != 0` 的样本求平均

这意味着：

- 最终均值的分母不一定是整个测试集
- 更像“所有 reward model 都成功打分的交集样本均值”

如果你开了 `UnifiedReward` 这种依赖服务的 judge，一旦它失败，一整条样本在最终均值里就会被排除。

所以 `MixGRPO` 的离线均值是“有条件的均值”，这点解读结果时一定要注意。

### 4.7 这些指标算不算“行业公认”

我会这样分：

| 指标 | 地位 |
| --- | --- |
| `HPSv2` | 很常见的图文偏好/对齐 proxy |
| `ImageReward` | 很常见的人类偏好代理模型 |
| `PickScore` | 很常见的人类偏好代理模型 |
| `CLIPScore` | 很常见的自动图文对齐 proxy |
| `UnifiedReward` | 新式 VLM judge，更像趋势型指标 |

但 `MixGRPO` 这套“用 HPDv2 test prompts 生成图，再让一组 reward models 分别打分”的整体评测方式，更像：

- 论文/项目内部常用的 reward-suite eval

而不是：

- 单一、稳定、公共 benchmark

所以我的结论是：

- 指标本身大多是社区熟悉的
- 但整个 eval protocol 更像“proxy 评测套件”，不是像 `GenEval strict_accuracy` 那样的强 benchmark

---

## 5. TreeGRPO：怎么做 eval

### 5.1 先说最关键的事实

当前这个 `TreeGRPO-main` 仓库版本，没有看到独立测试集评测脚本。

它的核心代码更像“训练主循环 + 训练日志”版本，而不是“完整实验复现仓库 + 多 benchmark eval 脚本”版本。

### 5.2 为什么我这么说

从代码结构看：

- `dataset.py` 只读一个 `prompt_path`
- `configs/base.yaml` 默认就是 `./prompts.txt`
- `train.py` 里没有 `test_dataset`
- 没有单独的 `eval()` 函数
- 没有 `eval_reward.py` 之类的脚本

也就是说，它当前没有像 `flow_grpo` 那样的：

- `train split`
- `test split`
- 周期性 held-out eval

### 5.3 它训练时到底记录了什么

`train.py` 里的 `sample(...)` 会做这些事情：

1. 对当前 batch prompt 采样一棵树。
2. 找到所有叶子节点图像。
3. 用 `HPS_v2` 给每个叶子打分。
4. 用这些叶子 reward 算均值和标准差。
5. 把 `reward/mean`、`reward/std` 记到 tracker。

注意这里的 reward 来源是：

- 当前训练 batch 的 prompt
- 当前模型采样的树叶结果

所以它不是 held-out test score，而是 training-time monitor。

### 5.4 TreeGRPO 用的是什么评测指标

当前仓库的主 reward model 基本只有一个：

- `HPSv2`

`reward_models/hps.py` 的实现就是：

- 用 HPSv2 的 image/text encoder
- 算 `image_features @ text_features.T`

训练日志里最核心的质量指标只有：

- `reward/mean`
- `reward/std`

此外还有一批 PPO/GRPO 训练诊断项：

- `train/loss`
- `train/approx_kl`
- `train/clipfrac`
- `train/ratio_mean`
- `train/ratio_min`

这些都不是“图片质量 benchmark”，只是优化过程统计。

### 5.5 TreeGRPO 当前的“评测”更像什么

更准确地说，它更像：

- `online reward monitoring`

而不是：

- `offline benchmark evaluation`

如果你把它和 `flow_grpo` 对比：

- `flow_grpo`：有 test split eval
- `TreeGRPO`：没有

如果你把它和 `MixGRPO` 对比：

- `MixGRPO`：至少有单独离线打分脚本
- `TreeGRPO`：也没有

### 5.6 这套指标算不算“行业公认”

`HPSv2` 本身当然是社区里很常见的自动评测 / reward 模型。

但“只报 HPSv2 的 training reward mean/std”这件事，不算完整、强势的 benchmark 评测。

所以我对 `TreeGRPO` 当前仓库的判断是：

- 用的是公认的 proxy 模型
- 但没有给出公认的 eval protocol

换句话说，`metric` 还行，`evaluation setup` 不完整。

---

## 6. 三者横向对比

| 维度 | Flow-GRPO / GRPO-Guard | MixGRPO | TreeGRPO |
| --- | --- | --- | --- |
| 是否有独立测试集 eval | 有 | 有，但在训练脚本外 | 当前仓库没有 |
| eval 触发方式 | 训练中每隔 `eval_freq` 周期性执行 | 先离线生成，再单独执行 `eval_reward.py` | 无独立 eval |
| 训练中会不会看 reward | 会 | 会 | 会 |
| 训练中 reward 是不是正式 benchmark | 不是 | 不是 | 不是 |
| 最像 benchmark 的指标 | `GenEval accuracy / strict_accuracy` | 没有单一 benchmark，更多是多 reward-model 报告 | 没有 |
| 常见 proxy 指标 | PickScore / CLIPScore / ImageReward / OCR / UnifiedReward | HPSv2 / ImageReward / PickScore / CLIPScore / UnifiedReward | HPSv2 |
| 训练诊断指标 | 有 | 有 | 有 |
| 评测完整度 | 三者里最高 | 中等 | 最弱 |

---

## 7. 如果你要把它们的“评测可信度”排个层次

我会这么排：

### 第一档：更像 benchmark

- `GenEval strict_accuracy`
- `GenEval accuracy`

原因：

- 有明确 test split
- 是任务正确率语义
- 更适合跨方法横向比较

### 第二档：成熟 proxy

- `PickScore`
- `ImageReward`
- `HPSv2`
- `CLIPScore`

原因：

- 社区非常常见
- 对图文对齐/偏好相关问题有参考价值
- 但本质还是代理指标，不等于真正人评

### 第三档：任务专用或新型 judge

- `OCR reward`
- `UnifiedReward`

原因：

- `OCR reward` 很适合文字渲染，但不通用
- `UnifiedReward` 很有前景，但 judge 依赖强、稳定性与口径更受实现影响

### 第四档：纯训练诊断

- `loss`
- `kl`
- `clipfrac`
- `ratio_mean`
- `ratio_min`
- `zero_std_ratio`

这些只能说明训练行为，不能直接说明生成结果更好。

---

## 8. 最后给你的直白判断

如果你后面要写论文、做汇报，或者整理成对比表，我建议你这样表述最稳：

- `Flow-GRPO / GRPO-Guard`：有较完整的 held-out eval 机制；在 `GenEval` 任务下，`accuracy / strict_accuracy` 是最像正式 benchmark 的指标；其他如 `PickScore / OCR / CLIPScore` 主要是 proxy。
- `MixGRPO`：有离线 eval 流程，但本质还是“生成后再喂多个 reward model 打分”；更适合说成 reward-model based evaluation suite，而不是单一公共 benchmark。
- `TreeGRPO`：当前开源仓库更偏训练核心代码，没有完整独立 eval；目前最主要的是训练时 `HPSv2 reward/mean/std` 监控，证据力弱于前两个。

一句话总结：

- `flow_grpo` 最像“有正式评测体系”
- `mixgrpo` 最像“离线多 reward 模型打分体系”
- `treegrpo` 当前仓库最像“训练监控体系”

