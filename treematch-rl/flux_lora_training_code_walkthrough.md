# Flux-LoRA 8xH200 训练代码梳理

这份文档专门解释当前 `treematch-rl` 仓库里，Flux LoRA 在单机 8 卡上是如何训练的。

它严格对应你现在这套代码和配置，而不是泛泛讲扩散模型训练流程。

本文主要对应这些文件：

- `scripts/train_flux.sh`
- `accelerate_configs/fsdp_small.yaml`
- `config/flux_lora.yaml`
- `treematch/train.py`
- `treematch/sampling/tree_sampler.py`
- `treematch/losses/*`
- `treematch/rewards/compute.py`

目标是把整条训练链条理清楚：

1. 启动时到底加载了什么
2. 8 张 H200 卡分别在做什么
3. 一个 prompt 是怎么变成 27 条路径的
4. reward 是怎么变成 loss 的
5. backward 和 optimizer.step 是怎么发生的
6. 你现在为了让 Flux LoRA 稳定训练，加了哪些机制


## 1. 你当前的训练环境在代码里是什么样

你现在的启动命令本质上是：

```bash
bash scripts/train_flux.sh config/flux_lora.yaml
```

`scripts/train_flux.sh` 里实际调用的是：

```bash
accelerate launch --config_file accelerate_configs/fsdp_small.yaml \
    --num_processes 8 \
    treematch/train.py --config "$CONFIG_PATH" "$@"
```

所以当前运行环境是：

- 单机
- 8 个进程
- 8 张 GPU
- `Accelerate + FSDP`
- 混合精度 `bf16`

`accelerate_configs/fsdp_small.yaml` 里最关键的设置有：

- `distributed_type: FSDP`
- `fsdp_sharding_strategy: SHARD_GRAD_OP`
- `fsdp_use_orig_params: true`
- `fsdp_activation_checkpointing: true`
- `fsdp_transformer_layer_cls_to_wrap: "FluxTransformerBlock,FluxSingleTransformerBlock"`

这在实际运行中表示：

- Flux transformer 会按 block 级别做 FSDP 包装
- 反向传播时用 FSDP 风格做梯度同步
- 激活采用 FSDP 级别的 activation checkpointing
- 代码刻意不再额外启用 HuggingFace 自己的 gradient checkpointing，避免双层重算带来的数值不稳定

这套配置就是为了在你这种 8xH200 的大卡环境里，把 Flux 训练尽量稳定地跑起来。


## 2. 你当前的 Flux LoRA 配置在代码里意味着什么

根据 `config/flux_lora.yaml`，你当前最关键的配置是：

- 基座模型：`flux.1-dev`
- LoRA rank：`64`
- LoRA alpha：`128`
- 分辨率：`512x512`
- 树分叉因子：`k=3`
- 总推理步数：`28`
- 固定分叉步：`[3, 8, 13]`
- 固定噪声强度：`[0.6, 0.9, 1.2]`
- reward 模型：`hpsv2 + clipscore`
- reward 融合方式：`advantage_aggr`
- beta warmup：`3 -> 5`，前 `20` 步
- `lambda_entropy = 1.0`
- `lambda_ref = 0.01`
- `ref_scale = 1.0`
- `is_clip_range = 0.2`
- `is_num_updates = 12`
- `learning_rate = 2e-4`
- `max_grad_norm = 20`
- `gradient_accumulation_steps = 1`
- `recompute_sub_batch = 4`
- `sanitize_lora_grad = true`
- `lora_grad_clip = 0.0`

这些不是“写在 yaml 里好看”的参数，它们都直接影响运行行为，比如：

- 一次 rollout 会采多少条路径
- reward target 分布有多尖锐
- 一次 rollout 会被重复利用多少次
- optimizer 更新有多激进
- recompute 时单次送入 transformer 的分支数
- LoRA 梯度在更新前会不会被清洗和截断


## 3. 整个训练流程的高层图景

每个全局训练步，大致做这些事情：

1. 每个 rank 从 dataloader 里取一个 prompt
2. 把 Flux text encoder 临时搬到 GPU
3. 把 prompt 编码成 Flux 所需的文本条件
4. 从一个 latent root 开始，采成 27 条树状路径
5. 把 27 个最终 latent 用 VAE 解码成图像
6. 用 reward model 给这 27 张图打分
7. 用当前策略重新计算这些路径的 path log-prob
8. 组装 `Softmax-TB + entropy + ref`
9. 反向传播到 LoRA 参数
10. FSDP 同步梯度
11. optimizer.step 和 lr_scheduler.step

最重要的理解是：

- rollout 本身大部分是在 `torch.no_grad()` 下做的
- 真正的训练梯度来自后面的 recompute
- 代码不是对原始整棵树直接反传
- 而是先存树的关键状态，再在训练阶段重算 SDE 路径 log-prob


## 4. 启动时到底加载了什么

在 `treematch/train.py` 的 `main()` 里，启动阶段大致分成这些步骤。

### 4.1 先读 YAML，再用 CLI 覆盖

代码先读 `--config` 指向的 yaml，然后把命令行参数覆盖进去。

所以你平时这样传：

```bash
--max_steps 1 --grad_accum 1 --is_num_updates 1 --recompute_sub_batch 2
```

是真的会改运行时行为，不只是打印出来。

### 4.2 创建 Accelerator

代码里创建的是：

```python
accelerator = Accelerator(mixed_precision=train_cfg.get("mixed_precision", "bf16"))
```

这里有个很重要的设计：

- 没有把 `gradient_accumulation_steps` 传给 `Accelerator`
- 梯度累积是手工靠 `transformer.no_sync()` 控制的

原因是：

- 这样可以把 FSDP 梯度同步时机握在代码自己手里
- 避免 `Accelerate` 内部累积计数和 FSDP 同步时序冲突

### 4.3 设置随机种子

代码里会调用：

```python
set_seed(seed)
```

这会初始化当前进程的随机数状态，帮助复现。

但要注意：

- 当前代码没有显式创建 `seed + rank` 的 per-rank `torch.Generator`
- 所以跨 rank 的 root noise 不是通过“显式 per-rank 生成器设计”来保证独立的

也就是说：

- 代码里有随机性
- 但没有专门设计成“每张卡的根噪声生成器完全独立可控”

### 4.4 加载 Flux pipeline

由于你的 `pretrained_path` 包含 `flux`，代码走的是：

```python
pipeline = FluxPipeline.from_pretrained(...)
```

然后取出：

- `transformer = pipeline.transformer`
- `vae = pipeline.vae`

当前处理方式是：

- `vae.requires_grad_(False)`
- VAE 放到 GPU
- text encoder 系列先冻结并放到 CPU

为什么 text encoder 平时放 CPU：

- 为了省显存
- 代码注释里明确写了，如果把 T5 / CLIP 一直留在 GPU，上大模型训练时很容易和 FSDP + activation + recompute 的开销叠在一起导致 OOM

### 4.5 应用 LoRA

你的当前配置启用了 LoRA，所以代码会构造：

- `LoraConfig(r=64, lora_alpha=128, target_modules=[注意力 + to_add_out + ff.net.* + ff_context.net.*])`

然后用 PEFT 把 LoRA 挂到 transformer 上。

接着，整个 transformer 连同 LoRA 一起转成 `bf16`：

```python
transformer = transformer.to(torch.bfloat16)
```

这么做的原因是：

- FSDP flatten 的一个参数组里要求 dtype 一致
- 如果 base 是 bf16、LoRA 是 fp32，会在 `accelerator.prepare()` 阶段出错

代码还做了一个 LoRA 初始化自检：

- 如果发现 trainable LoRA 参数里有 NaN / Inf，就做清理
- 如果没有，就正常继续训练

### 4.6 构建 optimizer 和 lr scheduler

当前 optimizer 是：

```python
torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad], ...)
```

因为现在是 LoRA 训练，所以：

- 只有 LoRA 参数在优化器里
- 巨大的 Flux base model 本体不参与更新

lr scheduler 是 cosine annealing。


## 5. 8 张卡是怎么拿数据的

数据集路径来自：

- `dataset.data_json_path = /home/zhaxianyu/lijiaming08/prompts_2k.jsonl`

`PromptDataset` 每次返回的样子是：

```python
{"prompt": prompt_text}
```

然后代码构造：

```python
DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
```

之后再进：

```python
transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(...)
```

在你这个 8 卡场景里，直观上可以这样理解：

- 每个 rank 一般会拿到不同 prompt
- 每个 rank 的 batch size 是 1
- 所以全局吞吐大致是“一次并行处理 8 个 prompt”

但要特别注意：

- 一个 rank 一次处理 1 个 prompt
- 这个 prompt 在该 rank 上会展开成 27 条 branch

所以当前训练不是“每张卡 27 个 prompt”，而是：

- 每张卡 1 个 prompt
- 每个 prompt 采 27 条路径


## 6. 从 prompt 到 Flux 条件输入

在 `train_one_step()` 里，选好 schedule 后，代码会先做 prompt 编码。

对于 Flux，代码调用：

```python
prompt_embeds = pipeline.encode_prompt(
    prompt=prompt,
    prompt_2=prompt,
    device=device,
    num_images_per_prompt=1,
    max_sequence_length=512,
)
```

得到的主要结果是：

- `encoder_hidden_states`
- `pooled_prompt_embeds`

然后还会构造：

- `text_ids`
- `latent_image_ids`

这是为了适配 Flux transformer 的 packed latent 输入格式。

这里一个非常关键的显存行为是：

1. text encoder 从 CPU 搬到 GPU
2. 编码 prompt
3. 编码完立即移回 CPU
4. 调用 `torch.cuda.empty_cache()`

这是你当前代码里最重要的省显存机制之一。


## 7. 一个 prompt 是怎么变成 27 条路径的

树采样发生在 `tree_sampler.sample()` 里。

### 7.1 根噪声

最初的 latent 是这样创建的：

```python
z_init = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
active_branches = [Branch(z_init)]
```

这意味着：

- 对于某张卡上的某个 prompt，整棵树从一个 root latent 开始
- 后面的所有 branch 都共享这个根起点

所以：

- 同一个 prompt 的 27 条路径不是 27 个不同根噪声起点
- 而是 1 个共同 root，后面在分叉步上再加噪声分开

### 7.2 Flux 前向是怎么跑的

对于 Flux，latent 会先从 `(B, C, H, W)` pack 成 token 形式。

forward 里会用到：

- `hidden_states = packed latent`
- `timestep = sigma`
- `encoder_hidden_states`
- `pooled_prompt_embeds`
- `txt_ids`
- `img_ids`
- `guidance = 3.5`

代码里还特意保持：

- `timestep` 用 fp32
- `guidance` 用 fp32
- transformer 前向主体在 bf16 autocast 下跑

### 7.3 为什么是 27 条

你当前配置是：

- `k = 3`
- 一共 3 次分叉

所以最终叶子数就是：

```text
3^3 = 27
```

当前固定分叉步是：

- 7
- 14
- 21

在这些步上会执行 SDE 分叉，噪声系数分别是：

- `0.4`
- `0.7`
- `1.0`

其余步要么是：

- ODE Euler 步
- 要么是压缩尾段里的 DPM-Solver++ 步

### 7.4 rollout 里都存了什么

每条 branch 会保留这些信息：

- 最终 latent
- 路径累计 log_prob
- 每个 SDE 步的 log_prob
- 每个 SDE 步的 mean
- 每个 SDE 步的输出 latent
- 每次分叉前的 latent 快照
- 对应的 sigma 历史

后面的训练阶段会用这些中间状态来重算当前策略下的 path log-prob。


## 8. DPM Flash 在这里是干什么的

你当前配置开了：

- `dpm_flash.enabled = true`
- `compress_ratio = 0.4`
- `solver_order = 2`
- `solver_type = midpoint`

这意味着采样后半段会被 DPM-Solver++ 压缩。

这么做的原因是：

- 后段去噪通常更接近确定性微调
- 用高阶 solver 替代很多尾部步，可以明显降低计算量
- 这样 27 branch 的 rollout 才更容易在大模型上跑得动

所以你的树采样并不是“28 步全都又贵又随机”，而是：

- 关键位置做 SDE 分叉
- 其他地方尽量用 ODE / DPM 降成本


## 9. VAE 解码和 reward 计算

rollout 结束后，这个 rank 上 27 条路径的最终 latent 会先拼起来。

然后做：

1. VAE 分批解码
2. 转成 PIL 图像
3. reward model 打分

你当前启用的 reward model 是：

- `hpsv2`
- `clipscore`

reward 计算用了线程池并行。

当前 reward 融合方式是：

- `mix_strategy: advantage_aggr`

它的含义是：

- 先对每个 reward model 的组内分数做 z-score
- 再按权重加起来

所以最终喂给 Softmax-TB 的 reward，不是原始加权和，而是“每个 reward model 先标准化后的加权结果”。

这里还有一个跟你当前环境很相关的现实点：

- Flux 编码 prompt 时最长可以走到 512 token
- 但 CLIP 类 reward 模型通常只能处理到 77 token 左右

所以你日志里看到的长 prompt truncation warning，是 reward 侧的问题，不是 Flux 生成侧的问题。


## 10. old policy、ref、entropy 在当前代码里分别是什么

reward 算完以后，代码会继续准备损失输入。

### 10.1 old log-probs

rollout 阶段路径 log_prob 会被收集成：

```python
old_log_probs = torch.tensor([b["log_prob_sum"] for b in branches], ...)
```

这表示：

- 当前这轮采样时，旧策略产生这些路径的 log-prob

### 10.2 ref log-probs

当前代码直接做了：

```python
ref_log_probs = old_log_probs.clone().detach()
```

所以当前 `ref` 不是“外部冻结参考模型”。
它更像是：

- 用本轮 rollout 的旧策略当锚点
- 起到 trust-region 风格的约束作用

### 10.3 entropy features

代码先通过空间平均池化提特征：

```python
path_features = ParticleEntropyLoss.compute_latent_features(all_latents.float())
```

但有一个当前实现上的重要细节：

真正传进 loss 的是：

```python
path_features=path_features.detach()
```

所以 entropy 这项现在会：

- 进入 loss 数值
- 进入日志

但它不是一个完整意义上、沿着生成图一路反传回去的 latent regularizer。


## 11. 真正产生梯度的是 recompute 阶段

这是当前训练最关键的地方。

因为 rollout 大部分是在 `torch.no_grad()` 下完成的，所以不能直接沿着原始整棵树反传。

训练真正拿梯度的方式是：

- 在 `recompute_path_log_probs()` 里，重新拿保存下来的 SDE 关键状态
- 用当前策略把这些路径对应的 log-prob 重新算一遍

对于每个 SDE 分叉步，代码会取出：

- `latent_in`
- `latent_out`
- old step log-prob
- old step mean

然后再跑当前 Flux transformer，得到：

- 当前 step log-prob
- 当前 step mean

最后：

```python
path_log_probs = sum(per_step_log_probs)
```

所以当前训练里真正可导的 path log-prob，不是完整 28 步全链条，而是：

- 3 个 SDE 分叉步对应 log-prob 的和

这也是为什么你现在很多 KL 风格指标会很小：

- 只统计 3 个随机决策点
- 每个 log-prob 本身还是按 latent 维度 mean 过的


## 12. 你当前代码里总损失是怎么组成的

当前总损失可以概括成：

```text
total_loss = weighted_tb + lambda_entropy * entropy + lambda_ref * ref
```

中间还夹着一些缩放和统计细节。

### 12.1 Softmax-TB

这是当前最核心的项。

代码实际做的是：

```text
log_p = log_softmax(path_log_probs)
log_r = log_softmax(beta * rewards)
per_path_tb = (log_p - log_r)^2
weighted_tb = mean(IS_weight * per_path_tb)
```

这就是你论文里“路径概率匹配奖励分布”的主要实现。

### 12.2 RatioNorm 重要性采样

因为你一次 rollout 后会做多次 inner update，代码会用这些量构造 IS 权重：

- 当前 step log-probs
- 旧 step log-probs
- 当前 step means
- 旧 step means
- 每步噪声尺度

你当前配置是：

- `is_clip_range = 0.2`
- `is_num_updates = 12`

也就是：

- 一次 rollout，最多会被重复用于 12 次优化更新

### 12.3 Entropy

entropy 是对最终 latent feature 做 RBF 排斥。

你当前的配置是：

- `lambda_entropy = 1.0`
- `rbf_bandwidth = 50.0`

### 12.4 Ref

当前 ref 的核心形式是：

```text
mean(((current_log_probs - ref_log_probs) / num_sde_steps)^2)
```

然后还会乘上：

- `ref_scale = 1.0`
- `lambda_ref = 0.01`

所以在你当前配置下，ref 仍然保留“别离 rollout 旧策略太远”的约束，但强度已经降到了更接近辅助正则的量级。


## 13. backward 在 8 张卡上到底是怎么发生的

在每个全局步内部，代码可能会做多个 inner update。

你当前配置里：

- `gradient_accumulation_steps = 1`
- `is_num_updates = 12`

因为 grad accumulation 是 1，所以每次 inner update 也都是一个实际可 step 的点。

反向更新流程大致是：

1. 计算 loss
2. 乘上 `backprop_scale / grad_accum`
3. `accelerator.backward(...)`
4. clip gradients
5. `optimizer.step()`
6. `lr_scheduler.step()`
7. `optimizer.zero_grad()`

如果 grad accumulation 大于 1，代码会对非最后一次 backward 使用：

- `transformer.no_sync()`

这是为了避免 FSDP 在每次 backward 都做 reduce-scatter，导致梯度被反复放大、最后在 bf16 下溢出成 NaN。

虽然你现在配置是 `grad_accum = 1`，但这套保护机制依然在代码里，是你这套训练能保持稳定的一个重要基础。


## 14. “8 张卡随着时间是怎么更新的”最直观的理解

当前最正确的脑图是：

在某个 global step：

- rank0 取一个 prompt
- rank1 取另一个 prompt
- ...
- rank7 取另一个 prompt

每个 rank 都独立执行：

1. 编码自己的 prompt
2. 采自己的 27 branch tree
3. 解自己的 27 个最终 latent
4. 算自己的 rewards
5. 重算自己的 path log-probs
6. 算自己的 loss
7. 在本地反传

然后在 backward / optimizer 更新阶段，通过 FSDP 把梯度同步起来。

所以你要这样理解：

- 8 张卡不是轮流工作
- 而是并行处理不同 prompt
- 然后在反向阶段同步参数更新

还有一个容易忽略的细节：

- prompt 数据是通过 `accelerator.prepare(dataloader)` 被分到不同 rank 的
- 但 root latent noise 不是靠显式的 per-rank generator 设计出来的

所以跨 rank 的随机性在实践中是存在的，但代码没有把它表达成一个非常明确的“每张卡固定用自己的生成器”机制。


## 15. 你现在为了让 Flux LoRA 稳定训练，加了哪些机制

这一部分是最重要的工程稳定性内容。

### 15.1 Flux block 级 FSDP 包装

`fsdp_small.yaml` 明确把这两类 block 都单独 wrap：

- `FluxTransformerBlock`
- `FluxSingleTransformerBlock`

这样做的目的，是避免整个 Flux 顶层被包成一个巨大的 FSDP 单元，导致 activation checkpointing 失效或者显存峰值过高。

### 15.2 只用 FSDP activation checkpointing

代码刻意不再同时启用：

```python
transformer.enable_gradient_checkpointing()
```

原因是：

- 双重 checkpointing 在这条 Flux + FSDP 训练链上容易把 backward 时序搞坏
- 进而导致梯度 corruption 或 NaN

### 15.3 text encoder 常驻 CPU

text encoder 只有在 prompt 编码时才临时搬上 GPU，用完立刻下去。

这能明显节省大模型训练时的 VRAM。

### 15.4 LoRA 和 transformer 统一 dtype

LoRA 和 transformer 一起转成 bf16，避免 FSDP flatten 时 dtype 不一致。

### 15.5 LoRA 梯度清洗

如果 `sanitize_lora_grad = true`，LoRA A/B 的梯度会被：

- `nan_to_num`
- 在 `lora_grad_clip > 0` 时再 clamp 到固定范围

你当前是：

- `lora_grad_clip = 0.0`

也就是当前只保留非有限值清洗，不再做逐元素截断；真正的梯度约束交给 `max_grad_norm`。

### 15.6 prompt embedding 的有限值防护

如果 `encoder_hidden_states` 或 `pooled_prompt_embeds` 含 NaN / Inf，代码会先做清洗和截断，再进入 rollout。

### 15.7 recompute 子批次

`recompute_sub_batch = 4` 表示 recompute 时不会一次把 27 条路径全塞进 Flux transformer，而是分块送。

这是当前最主要的抗 OOM 机制之一。

### 15.8 recompute OOM 自动降批重试

如果 recompute 仍然 OOM：

- 代码会把 chunk size 减半
- 所有 rank 一起同步这个回退行为
- 然后重试

这样能避免多卡因为某一张卡先 OOM 导致执行路径不同步。

### 15.9 rollout / recompute 中的非有限值保护

在 rollout 和 recompute 阶段，代码对这些量都做了保护：

- latent
- model prediction
- mean
- log-prob

这是因为 Flux bf16 训练在少数 kernel / 输入组合下，确实可能瞬间产出 NaN / Inf。

### 15.10 Flux guidance 一致性

当前代码已经统一了：

- rollout 用的 guidance_scale
- recompute 用的 guidance_scale

对你当前 Flux 配置来说，就是：

- `guidance_scale = 3.5`

这一点很关键，因为如果 rollout 和 recompute 的 guidance 不一致，训练目标本身就会被破坏。

### 15.11 TF32 和 cuDNN safe mode

如果配置允许：

- TF32 会打开
- `cudnn.benchmark` 会关掉
- 一些已知但可恢复的 cuDNN warning 会被过滤

这不会改变数学目标，但能让训练更稳、日志也更干净。


## 16. 当前最关键的参数，在代码里到底怎么生效

下面只列最重要的一批。

### Model 和 LoRA

- `model.pretrained_path`
  - 决定走 Flux 还是 SD3 的加载逻辑
- `model.guidance_scale = 3.5`
  - 在 Flux rollout 和 recompute 中都参与 forward
- `model.lora.rank = 64`
  - 决定 LoRA 低秩维度
- `model.lora.alpha = 128`
  - 决定 LoRA 的有效缩放

### Tree 采样

- `tree.k = 3`
  - 对应最终 `3^3 = 27` 条路径
- `tree.num_inference_steps = 28`
  - 整条 denoise schedule 的长度
- `tree.fixed_split_steps = [3, 8, 13]`
  - 3 个分叉位置
- `tree.fixed_noise_levels = [0.6, 0.9, 1.2]`
  - 3 次分叉时的噪声强度
- `tree.shift = 3.0`
  - sigma schedule 的 warp 方式

### DPM 尾段压缩

- `dpm_flash.enabled = true`
- `compress_ratio = 0.4`
- `solver_order = 2`
- `solver_type = midpoint`

这些参数一起控制尾段 DPM 加速。

### Reward

- `reward.models = [hpsv2, clipscore]`
  - 会同时构建两个 reward model
- `reward.weights = [0.6, 0.4]`
  - 融合权重
- `reward.mix_strategy = advantage_aggr`
  - 先逐模型 z-score，再加权

### Softmax-TB 和正则

- `loss.beta = 3.0`
- `loss.beta_target = 5.0`
- `loss.beta_warmup_steps = 20`
  - 前 20 步里 target 分布会逐步升温, 但不会像 `40.0` 那样接近 one-hot

- `loss.lambda_entropy = 1.0`
- `loss.rbf_bandwidth = 50.0`
  - 控制 entropy 的强度和作用半径

- `loss.lambda_ref = 0.01`
- `loss.ref_scale = 1.0`
  - 当前把 ref 降到辅助约束量级，避免持续把当前策略拉回 rollout 旧策略

- `loss.is_clip_range = 0.2`
- `loss.is_num_updates = 12`
  - 控制一次 rollout 后重复做多少次 off-policy 更新

### Training

- `training.learning_rate = 2e-4`
  - 对 LoRA 来说已经比较激进
- `training.max_grad_norm = 20.0`
  - 梯度裁剪阈值
- `training.gradient_accumulation_steps = 1`
  - 每次 inner update 都可以直接 step
- `training.recompute_sub_batch = 4`
  - recompute 的显存控制
- `training.sanitize_lora_grad = true`
  - 保留 LoRA 梯度中的 NaN/Inf 清洗
- `training.lora_grad_clip = 0.0`
  - 不再做逐元素 clamp，避免把有效梯度方向切碎


## 17. 你现在常见日志在代码里分别是什么意思

如果你看到：

```text
[Step N] Sampled 27 branches
```

表示的是：

- 当前这个 rank 上，这个 prompt 成功采出了 27 条叶子路径

如果你看到：

```text
loss=...
tb=...
entropy=...
ref=...
w_ent=...
w_ref=...
```

它们分别表示：

- `loss`：总损失
- `tb`：Softmax-TB 主项
- `entropy`：原始 entropy 项
- `ref`：经过 `ref_scale` 后的 ref 项
- `w_ent`：`lambda_entropy * entropy`
- `w_ref`：`lambda_ref * ref`

如果你看到：

```text
approx_kl=...
ratio=...
clipfrac=...
```

一定要记住：

- 这里的 `approx_kl` 不是完整 PPO 式策略 KL
- 它只是当前代码里一个比较压缩的 path log-prob 诊断量

如果你看到：

```text
grad_norm=...
grad_clip_coef=...
```

当前实现里要特别小心：

- `grad_norm` 打印的是全 rank 里最大的那个值
- `grad_clip_coef` 却是当前本地 rank 算出来的系数

所以它俩不是严格一一对应的一对日志值。


## 18. 当前代码里最容易误解的几个点

### 18.1 entropy 现在不是完整可导的 latent regularizer

因为 `path_features.detach()` 被传进 loss，所以 entropy 现在是“数值上参与 total loss”，但不是沿着生成图完整反传回去的那种实现。

### 18.2 ref 不是外部冻结 reference model

`ref_log_probs` 现在就是本轮 rollout 的 `old_log_probs`，所以它更像 trust-region anchor，而不是一个独立 reference policy。

### 18.3 reward 侧会截断长 prompt

Flux 生成能看更长 prompt，但 CLIP 类 reward model 只能看短很多的 token 长度，所以 reward 看到的文本不一定是完整 prompt。

### 18.4 跨 rank root noise 不是显式设计成独立 generator 的

当前是依赖进程自己的 RNG 状态，而不是明确写出一个 per-rank root noise 生成器。


## 19. 用一句话概括你当前这套 Flux LoRA 训练

最简洁但正确的脑图是：

1. 8 个 rank 用 FSDP 启动
2. 每个 rank 取 1 个 prompt
3. text encoder 临时上 GPU 编码 prompt，然后回 CPU
4. 每个 rank 上的 1 个 root latent 被扩成 27 条树状路径
5. 27 个最终 latent 被解码并用 HPS + CLIPScore 打分
6. reward 用 `advantage_aggr` 融合
7. 保存下来的 SDE 分叉状态被当前策略重新 replay
8. 当前 path log-prob 与 reward 诱导出的目标分布做 Softmax-TB 匹配
9. 再叠加 ref 和 entropy
10. backward 只更新 LoRA 参数
11. FSDP 同步分布式梯度
12. optimizer.step，周期性保存 checkpoint 和分布式 state


## 20. 如果你想最快看懂这个仓库，推荐阅读顺序

推荐按这个顺序读：

1. `scripts/train_flux.sh`
2. `accelerate_configs/fsdp_small.yaml`
3. `config/flux_lora.yaml`
4. `treematch/train.py`
5. `treematch/sampling/tree_sampler.py`
6. `treematch/sampling/sde_step.py`
7. `treematch/rewards/compute.py`
8. `treematch/losses/softmax_tb.py`
9. `treematch/losses/total_loss.py`
10. `treematch/utils/checkpoint.py`

这个顺序和代码真实执行顺序已经很接近了，最适合拿来理当前训练逻辑。
