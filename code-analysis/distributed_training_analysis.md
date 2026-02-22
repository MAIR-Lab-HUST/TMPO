# Diffusion RL 项目分布式训练框架深度分析

> 本文档深入分析 TreeGRPO、flow_grpo、FlowRL 三个项目使用的代码框架、GPU 并行计算策略、分布式训练方案和显存优化技术。

---

## 一、三个项目的框架选型对比

```mermaid
graph LR
    subgraph "TreeGRPO"
        A1["HuggingFace Accelerate<br/>DDP 数据并行"] --> A2["单节点多卡"]
        A2 --> A3["accelerate launch<br/>--num_processes=8"]
    end
    
    subgraph "flow_grpo"
        B1["HuggingFace Accelerate<br/>+ PyTorch FSDP<br/>+ DeepSpeed ZeRO"] --> B2["单节点 / 多节点"]
        B2 --> B3["5种配置灵活切换"]
    end
    
    subgraph "FlowRL"
        C1["verl 框架<br/>Ray + FSDP + vLLM"] --> C2["多节点异步"]
        C2 --> C3["python3 -m verl.trainer.main_ppo"]
    end
```

| 维度 | TreeGRPO | flow_grpo | FlowRL |
|------|----------|-----------|--------|
| **核心框架** | Accelerate (DDP) | Accelerate + FSDP/DeepSpeed | verl (Ray + FSDP + vLLM) |
| **并行策略** | 数据并行 DDP | 数据并行 + 模型分片 | Actor-Critic 分离调度 |
| **典型 GPU 数** | 1-8 卡 | 1-32 卡 | 8-64 卡 |
| **多节点** | 不支持 | 支持（NCCL） | 原生支持（Ray） |
| **显存优化** | 梯度检查点 | FSDP分片+CPU卸载+梯度检查点 | FSDP+vLLM+tensor并行 |
| **模型分片** | 无 | FULL_SHARD | FULL_SHARD |
| **推理引擎** | diffusers 原生 | diffusers 原生 | vLLM 独立进程 |
| **适用模型** | SD3.5-M | SD3/Flux/WAN | Qwen/DeepSeek LLM |

---

## 二、TreeGRPO：最简 Accelerate DDP

### 2.1 框架架构

TreeGRPO 使用最基础的 HuggingFace Accelerate，核心就是 **DDP（DistributedDataParallel）+ 混合精度**：

```python
# train.py 第 25-43 行
class RLTrainer:
    def __init__(self, config, accelerator, ...):
        self.accelerator = accelerator
        
        # 混合精度推理
        if accelerator.mixed_precision == "bf16":
            self.inference_dtype = torch.bfloat16
        
        # 模型加载到各自 GPU
        self.pipe = StableDiffusion3Pipeline.from_pretrained(...)
        self.pipe.transformer.enable_gradient_checkpointing()  # 显存优化
        self.pipe.transformer.requires_grad_(True)  # 全参数微调
```

### 2.2 启动方式

```bash
# README.md 记录的启动命令
accelerate launch --num_processes=8 train.py \
    training.epochs=300 \
    training.batch_size=1 \
    sample.num_trees=1
```

### 2.3 GPU 通信模式

```mermaid
graph TD
    subgraph "TreeGRPO DDP"
        GPU0["GPU 0<br/>完整模型副本<br/>Prompt A → 树采样"] 
        GPU1["GPU 1<br/>完整模型副本<br/>Prompt B → 树采样"]
        GPU7["GPU 7<br/>完整模型副本<br/>Prompt H → 树采样"]
        
        GPU0 -->|"AllReduce 梯度"| SYNC["梯度同步"]
        GPU1 -->|"AllReduce 梯度"| SYNC
        GPU7 -->|"AllReduce 梯度"| SYNC
    end
```

**设计特点**：
- 每张 GPU 持有**完整模型副本**，独立处理不同 prompt
- 每张 GPU 独立构建 $k^w = 16$ 张图的树
- 仅在反向传播时通过 AllReduce 同步梯度
- **优势**：代码极简，无复杂分片逻辑
- **劣势**：显存受限于单卡容量，无法训练超大模型

### 2.4 显存优化

TreeGRPO 仅使用了一种显存优化手段：

```python
# 梯度检查点（Activation Checkpointing）
self.pipe.transformer.enable_gradient_checkpointing()
```

以时间换空间：前向传播不保存中间激活值，反向传播时重新计算。

---

## 三、flow_grpo：全方位 Accelerate + FSDP + DeepSpeed

### 3.1 5种并行配置

flow_grpo 提供了 5 种灵活的 Accelerate 配置文件：

#### 配置 1: `multi_gpu.yaml` — 基础多 GPU DDP
```yaml
distributed_type: MULTI_GPU
mixed_precision: fp16
num_machines: 1
num_processes: 8          # 8 卡 DDP
```
最简单配置，每卡复制完整模型。

#### 配置 2: `fsdp.yaml` — PyTorch FSDP 全分片 ⭐
```yaml
distributed_type: FSDP
fsdp_config:
    fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP  # 按 Transformer 层分片
    fsdp_backward_prefetch: BACKWARD_PRE           # 反向传播预取下一层
    fsdp_forward_prefetch: true                    # 前向传播预取
    fsdp_offload_params: false                     # 不卸载到 CPU
    fsdp_sharding_strategy: FULL_SHARD             # 完全分片
    fsdp_use_orig_params: true                     # 保留原始参数名
    fsdp_activation_checkpointing: true            # 梯度检查点
mixed_precision: bf16
num_processes: 2
```

#### 配置 3: `deepspeed_zero2.yaml` — DeepSpeed ZeRO Stage 2
```yaml
distributed_type: DEEPSPEED
deepspeed_config:
    zero_stage: 2                   # 优化器状态 + 梯度分片
    offload_optimizer_device: none  # 不卸载优化器到 CPU
    offload_param_device: none      # 不卸载参数到 CPU
num_processes: 8
```

#### 配置 4: `multi_node.yaml` — 多节点
```yaml
distributed_type: MULTI_GPU
main_process_ip: '10.82.139.22'    # 主节点 IP
main_process_port: 19500
num_machines: 2
num_processes: 16                   # 2x8 GPU
rdzv_backend: static
```

#### 配置 5: `deepspeed_zero1.yaml` — ZeRO Stage 1
仅分片优化器状态，最低通信开销。

### 3.2 FSDP 核心代码 (`fsdp_utils.py`)

```python
class FSDPConfig:
    """FSDP 配置封装"""
    sharding_strategy = "FULL_SHARD"    # 完全分片策略
    backward_prefetch = "BACKWARD_PRE"  # 预取优化
    cpu_offload = False                 # CPU 卸载开关
    num_replicate = 1                   # Hybrid Shard 复制数
    num_shard = 8                       # 分片数
    mixed_precision_dtype = torch.bfloat16
    use_activation_checkpointing = True
    use_device_mesh = False             # 用于 Hybrid Shard 的 device mesh
```

**FSDP Wrapper 工作原理**：

```mermaid
graph TD
    subgraph "FSDP FULL_SHARD 模式"
        M["SD3 Transformer<br/>~2B 参数"]
        M --> L0["Layer 0<br/>分片到 GPU0-7"]
        M --> L1["Layer 1<br/>分片到 GPU0-7"]
        M --> LN["Layer N<br/>分片到 GPU0-7"]
    end
    
    subgraph "前向传播 GPU 0"
        L0 -->|"AllGather 汇聚完整层"| FW0["Layer 0 前向"]
        FW0 -->|"释放其他分片"| L1_FW["AllGather Layer 1"]
        L1_FW --> FWN["..."]
    end
    
    subgraph "反向传播 GPU 0" 
        BWN["...反向"] -->|"AllGather + ReduceScatter"| BW1["Layer 1 反向"]
        BW1 --> BW0["Layer 0 反向"]
    end
```

```python
def fsdp_wrapper(model, fsdp_config, get_transformer_layer_cls):
    """将模型包装为 FSDP"""
    
    # Hybrid Shard: 节点内分片，节点间复制
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD' and fsdp_config.use_device_mesh:
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(num_replicate, num_shard),  # 例如 (4节点, 8卡/节点)
            mesh_dim_names=("replicate", "shard")
        )
    
    fsdp_model = FSDP(
        model,
        # 按 Transformer 层自动分片
        auto_wrap_policy=transformer_auto_wrap_policy(
            transformer_layer_cls=get_transformer_layer_cls()
        ),
        mixed_precision=MixedPrecision(
            param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=False),
        use_orig_params=True,  # LoRA 兼容必须
    )
    
    # 梯度检查点
    if use_activation_checkpointing:
        apply_activation_checkpointing(fsdp_model, ...)
    
    return fsdp_model
```

### 3.3 优化器 CPU 卸载 (`OptimizerOffloadHook`)

flow_grpo 实现了一个精巧的**优化器状态 CPU 卸载**机制：

```python
class OptimizerOffloadHook:
    """优化器步进前后将状态在 GPU/CPU 间搬运"""
    
    def pre_step_hook(self, optimizer, args, kwargs):
        """优化器步进前：CPU → GPU"""
        for param in params:
            for key, cpu_tensor in self.cpu_states[param].items():
                state[key] = cpu_tensor.to(param.device, non_blocking=True)
    
    def post_step_hook(self, optimizer, args, kwargs):
        """优化器步进后：GPU → CPU"""
        for param in params:
            for key, tensor in state.items():
                self.cpu_states[param][key] = tensor.to('cpu', non_blocking=True)
                state[key] = torch.empty(0, device=param.device)  # 释放 GPU
```

**效果**：AdamW 的 momentum 和 variance 两个缓冲区（与参数同样大小）只在 optimizer.step() 时短暂加载到 GPU，其余时间存在 CPU 内存中。对于 2B 参数模型，可节省约 **16GB 显存**。

### 3.4 多节点启动脚本

```bash
# scripts/multi_node/sd3.sh
export NCCL_IB_DISABLE=0    # 启用 InfiniBand（高速互连）
export NCCL_IB_HCA=mlx5     # 指定 RDMA 设备
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3  # GID 索引

accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
    --num_machines 4 --num_processes 32 \             # 4节点 × 8卡
    --machine_rank ${RANK} \                          # 当前节点编号
    --main_process_ip ${MASTER_ADDR} \                # 主节点 IP
    --main_process_port ${MASTER_PORT} \
    scripts/train_sd3.py --config config/grpo.py:geneval_sd3
```

### 3.5 训练代码中的关键分布式操作

```python
# train_sd3.py 中的分布式关键点

# 1. Accelerator 初始化（核心！）
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    # 梯度累积 = 样本累积步 × 时间步数
    gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
)

# 2. 模型/优化器/数据包装
transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
    transformer, optimizer, train_dataloader, test_dataloader
)

# 3. K-Repeat 分布式采样器（每 GPU 不同 prompt）
train_sampler = DistributedKRepeatSampler(
    dataset=train_dataset,
    batch_size=config.sample.train_batch_size,
    k=config.sample.num_image_per_prompt,
    num_replicas=accelerator.num_processes,  # GPU 总数
    rank=accelerator.process_index,           # 当前 GPU 编号
    seed=42
)

# 4. 跨 GPU 汇聚奖励（AllGather）
gathered_rewards = {
    key: accelerator.gather(value)  # 每个 GPU 的奖励汇聚到所有 GPU
    for key, value in samples["rewards"].items()
}

# 5. Per-Prompt 归一化需要全局数据
# 先 gather 所有 GPU 的 prompt_ids，再解码
prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

# 6. 全局归一化后，按 GPU 编号取回本地优势
advantages = advantages.reshape(
    accelerator.num_processes, -1, advantages.shape[-1]
)[accelerator.process_index]

# 7. 梯度累积 + 同步
with accelerator.accumulate(transformer):  # 自动管理梯度累积
    loss = compute_loss(...)
    accelerator.backward(loss)             # 分布式反向
    if accelerator.sync_gradients:         # 梯度同步时裁剪
        accelerator.clip_grad_norm_(transformer.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
```

### 3.6 GPU 通信模式图

```mermaid
sequenceDiagram
    participant GPU0
    participant GPU1
    participant GPU7
    
    Note over GPU0,GPU7: === 采样阶段 (各自独立) ===
    GPU0->>GPU0: prompt_A → SDE采样 → 图像 → 奖励
    GPU1->>GPU1: prompt_B → SDE采样 → 图像 → 奖励
    GPU7->>GPU7: prompt_H → SDE采样 → 图像 → 奖励
    
    Note over GPU0,GPU7: === AllGather 奖励 ===
    GPU0-->>GPU1: gather rewards
    GPU1-->>GPU0: gather rewards
    GPU7-->>GPU0: gather rewards
    
    Note over GPU0,GPU7: === Per-Prompt 归一化 (每卡相同计算) ===
    GPU0->>GPU0: advantages = normalize(all_rewards)
    GPU1->>GPU1: advantages = normalize(all_rewards)
    GPU7->>GPU7: advantages = normalize(all_rewards)
    
    Note over GPU0,GPU7: === 训练阶段 (各自训练本地数据) ===
    GPU0->>GPU0: 遍历时间步 → PPO-Clip loss
    GPU1->>GPU1: 遍历时间步 → PPO-Clip loss
    GPU7->>GPU7: 遍历时间步 → PPO-Clip loss
    
    Note over GPU0,GPU7: === AllReduce 梯度 (FSDP/DDP) ===
    GPU0-->>GPU7: ReduceScatter gradients
    GPU7-->>GPU0: AllGather params
```

---

## 四、FlowRL：verl 框架 (Ray + FSDP + vLLM)

### 4.1 verl 架构概览

FlowRL 不直接使用 Accelerate，而是基于字节跳动的 **verl** 框架（RL for LLM 的专用框架）：

```mermaid
graph TD
    subgraph "verl 框架架构"
        RAY["Ray 集群调度器"] --> ACTOR["Actor Worker<br/>(FSDP 分片)"]
        RAY --> CRITIC["Critic Worker<br/>(FlowRL 不使用)"]  
        RAY --> REF["Reference Worker<br/>(FSDP, 冻结权重)"]
        RAY --> ROLLOUT["Rollout Worker<br/>(vLLM 推理)"]
        RAY --> REWARD["Reward Worker<br/>(奖励计算)"]
    end
    
    subgraph "训练流程"
        ROLLOUT -->|"生成 response"| REWARD
        REWARD -->|"reward"| ACTOR
        REF -->|"ref_log_prob"| ACTOR
        ACTOR -->|"更新权重"| ROLLOUT
    end
```

### 4.2 启动方式

```bash
# command/training/math/flowrl_7B_math.sh
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    \
    # Actor FSDP 配置
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    # Rollout vLLM 配置
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \    # tensor 并行
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \      # GPU 显存占比
    actor_rollout_ref.rollout.n=8 \                             # 每 prompt 生成 8 条回复
    \
    # Reference 配置
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    # 集群配置
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \          # 节点数
    trainer.total_epochs=1
```

### 4.3 Actor-Rollout-Reference 分离

verl 框架的核心设计是将**训练 (Actor)** 和**推理 (Rollout)** 分开：

```mermaid
graph LR
    subgraph "Rollout Phase (vLLM)"
        R1["GPU 0-7<br/>vLLM 推理引擎<br/>Tensor Parallel"]
        R1 -->|"batch=512<br/>n=8 per prompt"| RESP["4096 条 response"]
    end
    
    subgraph "Reward Phase"
        RESP --> RM["奖励计算<br/>math accuracy"]
        RM --> REWARDS["rewards (4096,)"]
    end
    
    subgraph "Training Phase (FSDP)"
        REWARDS --> A1["GPU 0-7<br/>Actor FSDP<br/>FlowRL TB 更新"]
        REF_W["GPU 0-7<br/>Reference FSDP<br/>(冻结)"] -->|"ref_log_prob"| A1
    end
    
    A1 -->|"同步权重到 vLLM"| R1
```

**关键区别**：
- **Rollout**（推理阶段）使用 **vLLM** —— 高效推理引擎，支持 PagedAttention、continuous batching
- **Training**（训练阶段）使用 **FSDP** —— 模型参数分片到多卡上
- 两者共享 GPU，但交替执行，通过 **权重同步** 衔接

### 4.4 权重同步机制

FlowRL 修改了 `fsdp_vllm.py`，在 FSDP 训练完一轮后将更新的权重同步到 vLLM：

```python
# verl/workers/sharding_manager/fsdp_vllm.py
# 训练后同步到 vLLM
loaded_params = model.load_weights(
    (name, param.to(device).full_tensor() 
     if isinstance(param, DTensor) else param)
    for name, param in updated_params.items()
    if not name.startswith("proj_z")  # 跳过 proj_z（vLLM 不需要）
)
```

### 4.5 FlowRL 特有的 FSDP 配置

```bash
# 训练脚本中的 FSDP 配置
actor_rollout_ref.actor.fsdp_config.param_offload=False    # 参数不卸载到 CPU
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False # 优化器不卸载
actor_rollout_ref.ref.fsdp_config.param_offload=False       # 参考模型也不卸载
actor_rollout_ref.model.enable_gradient_checkpointing=True  # 梯度检查点
```

---

## 五、显存占用分析

### 5.1 Diffusion Model (SD3.5-M ~2.5B params) 显存预算

| 组件 | 大小 | 优化后 |
|------|------|--------|
| Transformer 参数 (bf16) | ~5 GB | FSDP 分片: 5/N GB/卡 |
| VAE + Text Encoder (冻结) | ~3 GB | 3 GB/卡（不分片） |
| LoRA 可训练参数 | ~0.1 GB | 0.1 GB/卡 |
| AdamW 优化器状态 | ~0.4 GB | CPU卸载: 0 GB/卡 |
| 梯度 | ~0.2 GB | 0.2 GB/卡 |
| 激活值（无检查点） | ~15 GB | 梯度检查点: ~3 GB |
| 采样中间 latent | ~2 GB | 2 GB/卡 |
| 奖励模型 (CLIP/HPSv2) | ~2 GB | 2 GB/卡 |
| **总计 (单卡, 无优化)** | **~28 GB** | - |
| **总计 (8卡 FSDP + 优化)** | - | **~9 GB/卡** |

### 5.2 各项目显存优化策略对比

| 优化技术 | TreeGRPO | flow_grpo | FlowRL |
|---------|----------|-----------|--------|
| **LoRA 微调** | ❌ 全参数 | ✅ r=32 | ✅ LoRA |
| **梯度检查点** | ✅ | ✅ | ✅ |
| **FSDP 模型分片** | ❌ | ✅ FULL_SHARD | ✅ FULL_SHARD |
| **优化器 CPU 卸载** | ❌ | ✅ OptimizerOffloadHook | ❌ (可选) |
| **混合精度** | bf16 | fp16/bf16 | bf16 |
| **推理与训练分离** | ❌ | ❌ | ✅ vLLM |
| **Tensor 并行** | ❌ | ❌ | ✅ (vLLM rollout) |

---

## 六、DistributedKRepeatSampler — flow_grpo 的核心采样器

flow_grpo 设计了一个专门的分布式采样器来保证 Per-Prompt GRPO 的正确性：

```python
class DistributedKRepeatSampler:
    """分布式 K-重复采样器
    
    保证：同一个 prompt 生成的 K 张图片在同一张 GPU 上
    这样 Per-Prompt 归一化可以在本地计算
    """
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed):
        # batch_size: 每 GPU 的 prompt 数
        # k: 每个 prompt 生成的图片数
        # num_replicas: GPU 总数
        
        self.total_samples = batch_size * k * num_replicas
        self.m = self.total_samples // k  # 唯一 prompt 数
    
    def __iter__(self):
        # 1. 生成全局随机排列
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g)[:self.m]
        
        # 2. 每个 prompt 重复 k 次
        indices = indices.repeat_interleave(self.k)
        
        # 3. 按 GPU 切分（关键！同 prompt 在同 GPU）
        per_card = indices.view(self.num_replicas, -1)
        yield per_card[self.rank]
```

**设计意图**：GRPO 需要同一 prompt 的多张图像来计算组内优势。如果 K 张图分散在不同 GPU，就需要额外通信汇聚。通过这个采样器，同 prompt 的 K 张图保持在同一 GPU 上。

---

## 七、梯度累积的特殊处理

flow_grpo 中梯度累积的设计非常精妙：

```python
# 关键：gradient_accumulation_steps = 样本累积 × 时间步数
accelerator = Accelerator(
    gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps
)
```

**原因**：训练循环有双层嵌套——外层遍历样本，内层遍历时间步。每个时间步都做一次 `backward()`，但只有在完成所有时间步和指定的累积步数后才做 `optimizer.step()`。

```python
# 训练循环结构
for i, sample in enumerate(batches):           # 样本维度
    for j in range(num_train_timesteps):        # 时间步维度
        with accelerator.accumulate(transformer):  # 自动判断是否同步
            loss = compute_loss(j)
            accelerator.backward(loss)
            if accelerator.sync_gradients:      # 仅在累积完毕时
                clip_grad_norm_(...)
            optimizer.step()
            optimizer.zero_grad()
```

---

## 八、Z-AFT 推荐的分布式方案

基于以上分析，Z-AFT 项目推荐：

| 组件 | 方案 | 理由 |
|------|------|------|
| 框架 | HuggingFace Accelerate | 三个项目中最通用、最成熟 |
| 并行策略 | DDP (≤8卡) / FSDP (>8卡) | 按规模灵活切换 |
| 模型微调 | LoRA (r=32) | 显存友好，flow_grpo 验证有效 |
| 显存优化 | 梯度检查点 + 混合精度 bf16 | 基础必备 |
| 可选优化 | FSDP FULL_SHARD + Optimizer CPU Offload | 大模型 (Flux) 必需 |
| 采样器 | 自定义 TreeDistributedSampler | 保证同 prompt 的 27 分支在同 GPU |
| 梯度累积 | accumulate_steps × 分叉步数 | 适配树结构训练 |

```python
# Z-AFT 推荐的 Accelerator 初始化
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=gradient_accum * num_split_steps,  # 3 个分叉步
)

# Z-AFT 推荐的 FSDP 配置 (大模型场景)
# accelerate_configs/fsdp.yaml
distributed_type: FSDP
fsdp_config:
    fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
    fsdp_sharding_strategy: FULL_SHARD
    fsdp_activation_checkpointing: true
    fsdp_use_orig_params: true   # LoRA 兼容
mixed_precision: bf16
```
