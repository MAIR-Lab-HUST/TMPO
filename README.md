# TMPO

> **Tree-based Distribution Matching Policy Optimization for Diverse and Efficient Diffusion Model Alignment**

## Key Features

- **Softmax-TB Distribution Matching** — Replaces traditional reward maximization with GFlowNet trajectory balance for multi-modal coverage
- **3-level 27-branch Tree Sampling** — Prefix sharing reduces computation; only 3 SDE steps inject noise
- **Beta Distribution Adaptive Scheduling** — Dynamically adjusts split positions and noise levels based on online reward mean
- **RatioNorm IS** — Eliminates E[log w] < 0 bias in Flow Matching, supports multiple policy updates
- **8xH200 Optimized** — FSDP SHARD_GRAD_OP (ZeRO-2) + LoRA + bf16

---

## Project Structure

```
TMPO/
├── accelerate_configs/         # Accelerate distributed configs
│   └── fsdp_small.yaml         # FSDP FULL_SHARD
├── config/                     # Training configs (YAML)
│   ├── sd35m_lora.yaml         # SD3.5-Medium LoRA
│   └── flux_lora.yaml          # Flux LoRA
├── tmpo/                       # Core code
│   ├── train.py                # Main training entry + loop
│   ├── sampling/
│   │   ├── sde_step.py         # SDE noise stepping + log_prob
│   │   ├── scheduler.py        # Beta distribution adaptive scheduler
│   │   └── tree_sampler.py     # 3-level 27-branch tree sampler
│   ├── losses/
│   │   ├── softmax_tb.py       # Softmax-TB core loss
│   │   ├── ratio_norm.py       # RatioNorm IS weight computation
│   │   ├── entropy.py          # RBF particle entropy regularization
│   │   ├── reference.py        # Reference model constraint loss
│   │   └── total_loss.py       # Total loss assembly + diagnostics
│   ├── rewards/
│   │   ├── compute.py          # Multi-reward parallel compute + advantage_aggr fusion
│   │   ├── hpsv2.py            # HPSv2 (human preference score)
│   │   ├── clipscore.py        # CLIP Score (text-image alignment)
│   │   ├── pickscore.py        # PickScore (preference alignment)
│   │   └── geneval_http.py     # GenEval HTTP client
│   ├── eval/
│   │   ├── evaluator.py        # Inline evaluation during training
│   │   └── diversity.py        # L-GMD and cosine diversity metrics
│   └── utils/
│       ├── distributed.py      # AllGather + per-prompt normalization
│       ├── checkpoint.py       # LoRA / full-param checkpoint management
│       ├── ema.py              # EMA weight smoothing
│       └── logging_.py         # Logging utilities
├── scripts/
│   ├── train_flux.sh
│   └── setup_server.sh
└── data/
    └── pickapic_prompts.json   # Prompt data (user-provided)
```

---

## Environment Setup

```bash
# 1. Create conda environment
conda create -n tmpo python=3.10 -y
conda activate tmpo

# 2. Install PyTorch (CUDA 12.1)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install core dependencies
pip install accelerate==0.33.0 diffusers==0.30.0 transformers==4.44.0

# 4. Install LoRA + reward model dependencies
pip install peft open_clip_torch

# 5. Install utility libraries
pip install pyyaml Pillow numpy wandb

# 6. Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
"
```

---

## Dataset

Training uses a JSON-format prompt list. Each step samples one prompt and generates 27 paths for reward computation.

**Supported formats**:

```json
// Format A: plain list
["a photo of a cat", "a painting of mountains", ...]

// Format B: list of dicts
[{"prompt": "a photo of a cat"}, {"prompt": "a painting of mountains"}, ...]

// Format C: key-value pairs
{"001": "a photo of a cat", "002": "a painting of mountains", ...}
```

**Recommended source**: [Pick-a-Pic](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2) prompt subset.

Configuration:
```yaml
dataset:
  data_json_path: "data/pickapic_prompts.json"
  resolution: [512, 512]   # H x W; SD3.5: 512, Flux: 720
```

---

## Reward Models

TMPO uses **multi-reward parallel computation + advantage_aggr fusion**:

### Supported Reward Models

| Model | Dimension | Range | Weight |
|-------|-----------|-------|--------|
| **HPSv2** | Human preference (composition, detail, quality) | ~[0.2, 0.35] | 0.6 |
| **CLIP Score** | Text-image alignment (prompt matching) | ~[20, 35] | 0.4 |
| **PickScore** | Preference alignment | model-dependent | optional |
| **ImageReward** | Image-text quality | model-dependent | optional |
| **GenEval HTTP** | Compositional semantic rules | server-returned | optional |

### Fusion Strategy: `advantage_aggr` (recommended)

Each reward model is **independently normalized then weighted-summed**, resolving scale differences:

```
HPSv2:     [0.25, 0.28, 0.30, ...]  -> normalize -> [-1.2, 0.1, 1.1, ...]  x 0.6
CLIP:      [22.1, 25.3, 28.7, ...]  -> normalize -> [-1.0, 0.2, 1.5, ...]  x 0.4
                                                                              -----
Final advantage:                                                   weighted sum -> (K,)
```

### Reward Weight Preparation

```bash
mkdir -p reward_ckpt

# HPSv2 weights (required)
# Download HPS_v2.1_compressed.pt and open_clip_pytorch_model.bin to reward_ckpt/

# CLIP Score
# Config:
#   clip_score_model_path: "openai/clip-vit-large-patch14"

# PickScore (auto-downloaded from HuggingFace by default)
# Config:
#   pickscore_processor_path: "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
#   pickscore_model_path: "yuvalkirstain/PickScore_v1"

# ImageReward (explicit dependency install recommended)
pip install image-reward
pip install git+https://github.com/openai/CLIP.git

# GenEval HTTP (training side only sends requests)
# Config:
#   geneval_url: ""
# Note: you must start the reward server separately
```

### Reward Configuration Example

```yaml
reward:
    models: ["hpsv2", "pickscore", "imagereward", "geneval"]
    weights: [0.4, 0.2, 0.2, 0.2]
    mix_strategy: "advantage_aggr"

    hps_path: "./reward_ckpt/HPS_v2.1_compressed.pt"
    hps_clip_path: "./reward_ckpt/open_clip_pytorch_model.bin"
    clip_score_model_path: "./reward_ckpt/clip-vit-large-patch14"

    pickscore_processor_path: "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    pickscore_model_path: "yuvalkirstain/PickScore_v1"

    imagereward_model_name: "ImageReward-v1.0"
    imagereward_med_config: null

    geneval_url: ""
    geneval_batch_size: 64
    geneval_timeout: 120
    geneval_only_strict: true
    geneval_retries: 6
```

---

## Training

### Quick Start

```bash
# Default config (SD3.5-Medium LoRA)
bash scripts/train_flux.sh

# Manual launch
accelerate launch \
    --config_file accelerate_configs/fsdp_small.yaml \
    --num_processes 4 \
    tmpo/train.py \
    --config config/sd35m_lora.yaml \
    --lr 1e-5 --beta 15.0
```

### Distributed Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `fsdp_sharding_strategy` | `SHARD_GRAD_OP` | Gradient + optimizer state sharding, full model params retained |
| `fsdp_backward_prefetch` | `BACKWARD_PRE` | Backward prefetch for bandwidth utilization |
| `fsdp_activation_checkpointing` | `true` | Activation checkpointing to save memory |
| `mixed_precision` | `bf16` | Native bf16 support |
| `fsdp_use_orig_params` | `true` | Required for LoRA compatibility |

### Accelerate Launch Parameters

| Argument | Description | Example |
|----------|-------------|---------|
| `--config_file` | Accelerate config file | `accelerate_configs/fsdp_small.yaml` |
| `--num_processes` | Number of GPUs (overrides yaml) | `4` |
| `--mixed_precision` | Mixed precision type | `bf16` / `fp16` |
| `--main_process_port` | Main process port (avoid conflicts) | `29500` |
| `--gpu_ids` | Specify GPU IDs | `0,1,2,3` |

---

## CLI Parameters

All YAML config values can be overridden via CLI `--key value`. **CLI takes priority over YAML**.

### Training Parameters

| CLI | YAML Path | Default | Description |
|-----|-----------|---------|-------------|
| `--lr` | `training.learning_rate` | `1e-5` | Learning rate |
| `--max_steps` | `training.max_train_steps` | `300` | Max training steps |
| `--grad_accum` | `training.gradient_accumulation_steps` | `3` | Gradient accumulation steps |
| `--batch_size` | `training.vae_decode_batch_size` | `4` | VAE decode batch size |
| `--seed` | `training.seed` | `42` | Random seed |
| `--max_grad_norm` | `training.max_grad_norm` | `1.0` | Gradient clip norm |
| `--output_dir` | `training.output_dir` | `outputs/sd35m` | Output directory |

### Loss Parameters

| CLI | YAML Path | Default | Description |
|-----|-----------|---------|-------------|
| `--beta` | `loss.beta` | `15.0` | Softmax-TB temperature (higher = greedier) |
| `--lambda_entropy` | `loss.lambda_entropy` | `0.01` | Particle entropy regularization weight |
| `--lambda_ref` | `loss.lambda_ref` | `0.1` | Reference constraint weight |
| `--is_clip_range` | `loss.is_clip_range` | `0.2` | IS weight clip range |
| `--is_num_updates` | `loss.is_num_updates` | `4` | IS update iterations per sample |

> **Note**: GRPO-Guard's `beta` is a KL divergence coefficient (smaller = weaker constraint), while TMPO's `beta` is a Softmax-TB temperature (larger = more reward-greedy, 0 = uniform). They have entirely different semantics.

### Tree Parameters

| CLI | YAML Path | Default | Description |
|-----|-----------|---------|-------------|
| `--kappa` | `tree.kappa` | `4.0` | Beta distribution concentration (0 = uniform baseline) |
| `--num_inference_steps` | `tree.num_inference_steps` | `28` | Total sampling steps |
| `--tree_k` | `tree.k` | `3` | Branching factor per split (3 -> 27 paths) |

---

## Training Metrics

Each step prints two log lines:

```
[Step 1/300] loss=12.35 tb=11.23 entropy=0.08 ref=0.00 reward=0.25 alpha=0.50
             approx_kl=0.0023 clipfrac=0.12 ratio=1.002+/-0.045 log_prob=-1234.5 grad_norm=0.87
```

### Metric Interpretation

| Metric | Meaning | Healthy Range | Action on Anomaly |
|--------|---------|---------------|-------------------|
| **loss** | Total loss | Steadily decreasing | Diverging -> lower `--lr` |
| **tb** | Softmax-TB weighted loss | Steadily decreasing | Stalled -> adjust `--beta` |
| **entropy** | RBF particle entropy (path similarity) | Slowly decreasing | Not decreasing = good diversity |
| **ref** | Reference constraint loss | Low | Too high -> increase `--lambda_ref` |
| **reward** | Mean reward | Steadily increasing | Stalled -> check reward model paths |
| **alpha** | Difficulty level (Beta schedule) | 0->1 slowly | Always 0 = reward too low; always 1 = too high |
| **approx_kl** | Policy shift magnitude | < 0.1 | > 0.5 -> reduce `--is_num_updates` |
| **clipfrac** | IS weight clip fraction | < 0.3 | > 0.5 -> increase `--is_clip_range` |
| **ratio** | IS ratio mean +/- std | 1.0 +/- 0.1 | Far from 1.0 -> RatioNorm issue |
| **log_prob** | Current policy log_prob mean | Slowly changing | Sudden change -> model instability |
| **grad_norm** | Gradient norm (post-clip) | < `max_grad_norm` | Persistently clipped -> lower `--lr` |

---

## GPU Requirements

| Configuration | Minimum GPU | Recommended |
|---------------|-------------|-------------|
| SD3.5-M + LoRA | 1x24GB | 4x24GB (FSDP) |
| Flux + LoRA | 2x24GB | 4x80GB (FSDP) |
| SD3.5-M Full Params | 4x24GB | 8x80GB |
| **Large-scale (recommended)** | 4x80GB | **8xH200 80GB (SHARD_GRAD_OP)** |
