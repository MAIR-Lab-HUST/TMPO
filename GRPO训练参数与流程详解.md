# GRPO 核心实验参数对比（flow_grpo / GRPO-Guard / MixGRPO / TreeGRPO）

这份文档只回答三件事：

1. 数据集 prompt 大小（代码里如何定义）
2. 训练轮次如何定义（epoch 还是 step）
3. 关键训练参数怎么设计（采样、优化、调度）

---

## 1. 一页总览

| 方法 | 数据集 prompt 数（代码口径） | 训练轮次口径 | 默认训练长度（代码） | 每个 prompt 采样规模 | 备注 |
|---|---|---|---|---|---|
| flow_grpo | 未在配置中写死总数（由 dataset 下文件决定） | epoch 制 | num_epochs=100000（常手动早停） | 常见 num_image_per_prompt=24 | 采样器按 K-repeat 组织 |
| GRPO-Guard | 同 flow_grpo（不写死总数） | epoch 制 | 沿用 flow_grpo 外层循环 | 常见 num_image_per_prompt=24 | 核心改在 RatioNorm |
| MixGRPO | 不写死总数，来自 data_json_path | step 制为主 | max_train_steps=300（脚本默认） | num_generations=12（脚本常用） | 外层也有 epoch 循环，但停止主要看 step |
| TreeGRPO | 由 prompts.txt 决定；本仓库文件行为 103765 | epoch 制 | training.epochs=300 | 树叶子数 k^w（默认 16） | 每 prompt 训练样本是树节点数 |

说明：

- flow_grpo、GRPO-Guard、MixGRPO 在当前仓库里都没有把“prompt 总数”写成固定超参数，而是从数据文件读取。
- TreeGRPO 使用本地 prompts 文件，prompt 总量可直接从文件行数统计。

---

## 2. flow_grpo：参数如何设

## 2.1 数据集 prompt 数

- 配置里只指定 dataset 路径（如 dataset/pickscore、dataset/ocr、dataset/geneval）。
- 实际 prompt 总数取决于对应数据文件内容，不在 config/base.py 中写死。

## 2.2 训练轮次

- 外层是 epoch 循环。
- 默认上限：num_epochs=100000（工程实践通常按 reward 曲线提前停止）。

## 2.3 关键参数（常见 SD3 任务配置）

- sample.num_steps=10
- sample.eval_num_steps=40
- sample.guidance_scale=4.5
- sample.train_batch_size=9
- sample.num_image_per_prompt=24
- train.num_inner_epochs=1
- train.gradient_accumulation_steps=num_batches_per_epoch/2（常见设法）
- train.learning_rate 常见 1e-5（不同任务也有 1e-4）
- train.clip_range 常见 1e-4 或 1e-5
- train.beta 常见 0.01 到 0.04

## 2.4 step 如何拆分

- 一个 outer epoch：先采样，再训练。
- 一个 inner epoch：对采样结果按时间步做策略更新（如 10 个去噪步）。
- 每个 prompt 一次采样会扩成 K 条（K=num_image_per_prompt，常见 24）。

---

## 3. GRPO-Guard：参数如何设

## 3.1 数据集 prompt 数

- 与 flow_grpo 相同：由 dataset 文件决定，配置不写死总数。

## 3.2 训练轮次

- 仍是 epoch 制（外层组织方式与 flow_grpo 基本一致）。

## 3.3 关键参数（常见 guard 设定）

- sample.num_steps=10
- sample.num_image_per_prompt=24
- train.num_inner_epochs=1
- train.learning_rate 常见 1e-4
- train.beta 常见 0.0（不少 guard 配置关闭 KL）
- train.clip_range 常用极小值（如 2e-6、4e-6）
- rationorm=True

## 3.4 设计差异

- 采样组织基本不变。
- 核心变化在 loss：RatioNorm 修正 ratio 偏置，并统一不同时间步统计尺度。

---

## 4. MixGRPO：参数如何设

## 4.1 数据集 prompt 数

- 通过 data_json_path 指定（例如 data/rl_embeddings/prompt.json）。
- prompt 总数由 JSON 文件内容决定，不在 argparse 默认值中写死。

## 4.2 训练轮次

- 主停止条件是 max_train_steps（step 制）。
- 脚本有外层 epoch 循环，但实际实验通常按 step 预算和 checkpoint 频率管理。

## 4.3 关键参数（finetune 脚本常见）

- max_train_steps=300
- sampling_steps=25
- train_batch_size=1
- gradient_accumulation_steps=3
- learning_rate=1e-5
- clip_range=1e-4
- adv_clip_max=5.0
- num_generations=12
- timestep_fraction=0.6
- iters_per_group=25（Flash 常见 20）
- group_size=4
- sample_strategy=progressive
- kl_coeff=0.0
- dpm_post_compress_ratio=0.4

## 4.4 step 如何拆分

- 每个 train step 会先采样一批 prompt，再按所选时间步做 GRPO 更新。
- use_group 打开时，每个 prompt 会重复扩增 num_generations 次（常见 12）。
- progressive/group 相关参数控制“哪些时间步参与优化、多久滑动一次窗口”。

---

## 5. TreeGRPO：参数如何设

## 5.1 数据集 prompt 数

- 配置里 data.prompt_path=./prompts.txt。
- 本仓库该文件统计为 103765 行（即约 10 万级 prompt）。

## 5.2 训练轮次

- 标准 epoch 制。
- 默认 training.epochs=300。

## 5.3 关键参数（configs/base.yaml）

- training.epochs=300
- training.inner_epochs=1
- training.lr=1e-5
- training.clip_range=1e-4
- training.adv_clip_max=5
- training.gradient_accumulation_steps=8
- training.save_ckpt_every_epoch=20
- sample.num_inference_steps=10
- sample.num_prompts=1
- sample.num_trees=1
- sample.noise_level=0.7
- tree.w=4
- tree.k=2
- tree.s=1
- tree.tou=50

## 5.4 step 如何拆分

- 每个 epoch 抽取 sample.num_prompts 个 prompt（每进程），每个 prompt 构造 sample.num_trees 棵树。
- 每个 prompt 的叶子图像数是 k^w（默认 16）。
- 每个 prompt 的训练样本数是 (k^w-1)/(k-1)（默认 15）。
- 每隔 tou 个 epoch，树窗口向后移动 s 步。

---

## 6. 关键设计差异（实验设置视角）

1. prompt 总量定义方式
- flow_grpo / GRPO-Guard / MixGRPO：由外部数据文件决定，不写死。
- TreeGRPO：由 prompts.txt 直接控制，可直接数行。

2. 训练预算表达方式
- flow_grpo / GRPO-Guard / TreeGRPO：主要看 epoch。
- MixGRPO：主要看 max_train_steps。

3. 单 prompt 扩增方式
- flow_grpo / GRPO-Guard：K-repeat（常见 24）。
- MixGRPO：num_generations（常见 12）+ 组内调度。
- TreeGRPO：树分叉 k^w（默认 16）。

4. 时间步优化策略
- flow_grpo / GRPO-Guard：通常全步或窗口式变体。
- MixGRPO：窗口优化最明显（part/progressive）。
- TreeGRPO：由树窗口和 tou/s 控制。

---

## 7. 代码来源（用于核对）

- flow_grpo 默认配置：flow_grpo-main/config/base.py
- flow_grpo 任务配置：flow_grpo-main/config/grpo.py
- GRPO-Guard 配置：flow_grpo-main/config/grpo_guard.py
- flow_grpo 训练循环：flow_grpo-main/scripts/train_sd3_GRPO_Guard.py
- MixGRPO 训练主程序：MixGRPO-main/fastvideo/train_grpo_flux.py
- MixGRPO 启动脚本：MixGRPO-main/scripts/finetune/finetune_flux_grpo_MixGRPO.sh
- MixGRPO Flash 启动脚本：MixGRPO-main/scripts/finetune/finetune_flux_grpo_MixGRPO_Flash.sh
- TreeGRPO 配置：TreeGRPO-main/configs/base.yaml
- TreeGRPO prompt 文件：TreeGRPO-main/prompts.txt
