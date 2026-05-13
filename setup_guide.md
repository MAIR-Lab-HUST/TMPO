# TMPO Step-by-Step Deployment Guide

> Execute each step and verify before proceeding to the next.

---

## Step 1: Create conda environment

```bash
conda create -n tmpo python=3.10 -y
conda activate tmpo
```

**Verify**: `python --version` should show `3.10.x`

---

## Step 2: Install PyTorch

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

**Verify**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs {torch.cuda.device_count()}')"
```
Expected output: `PyTorch 2.3.0, CUDA 12.1, GPUs 8`

---

## Step 3: Install training dependencies

```bash
pip install accelerate==0.33.0
pip install diffusers==0.30.0
pip install transformers==4.44.0
pip install peft
```

**Verify**:
```bash
python -c "import accelerate, diffusers; print(f'Accelerate {accelerate.__version__}, Diffusers {diffusers.__version__}')"
```

---

## Step 4: Install reward model dependencies

```bash
pip install open_clip_torch
pip install requests
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
pip install pyyaml Pillow numpy
```

---

## Step 5: Install Wandb

```bash
pip install wandb
wandb login
# Enter your API key (from https://wandb.ai/authorize)
```

---

## Step 6: Login to HuggingFace

SD3.5-Medium requires accepting the License Agreement.

```bash
pip install huggingface_hub
huggingface-cli login
# Enter your HF token (from https://huggingface.co/settings/tokens)
```

Then visit https://huggingface.co/stabilityai/stable-diffusion-3.5-medium in your browser and click **Agree and access**.

**Verify**:
```bash
python -c "from huggingface_hub import HfFolder; print('Token:', 'OK' if HfFolder.get_token() else 'MISSING')"
```

---

## Step 7: Download SD3.5-Medium Model

Model is ~10GB, download to local `models/` directory:

```bash
mkdir -p models/sd35m

huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir ./models/sd35m
```

> If you get a permission error, visit https://huggingface.co/stabilityai/stable-diffusion-3.5-medium and click **Agree and access** first.

**Verify**: `ls models/sd35m/transformer/`

---

## Step 8: Download HPSv2 Reward Model Weights

```bash
mkdir -p reward_ckpt

# HPSv2 main weights
hf download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./reward_ckpt/
```

**Verify**: `ls -la reward_ckpt/HPS_v2.1_compressed.pt`

---

## Step 9: Download OpenCLIP Weights (HPSv2 backbone)

```bash
hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./reward_ckpt/
```

**Verify**: `ls -la reward_ckpt/open_clip_pytorch_model.bin`

---

## Step 10: CLIP Score Model

```bash
mkdir -p reward_ckpt/clip-vit-large-patch14

huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./reward_ckpt/clip-vit-large-patch14
```

**Verify**: `ls reward_ckpt/clip-vit-large-patch14/config.json`

---

## Step 10.1: PickScore (auto-downloaded by default)

On first training run with `pickscore`, these models are auto-downloaded from HuggingFace:

- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- `yuvalkirstain/PickScore_v1`

To pre-download locally:

```bash
mkdir -p reward_ckpt/pickscore_processor reward_ckpt/pickscore_model
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir ./reward_ckpt/pickscore_processor
huggingface-cli download yuvalkirstain/PickScore_v1 --local-dir ./reward_ckpt/pickscore_model
```

---

## Step 10.2: ImageReward Weights

ImageReward can auto-download using model name `ImageReward-v1.0`.

To use a fixed local path:

```bash
mkdir -p reward_ckpt/image_reward
huggingface-cli download THUDM/ImageReward ImageReward.pt --local-dir ./reward_ckpt/image_reward
huggingface-cli download THUDM/ImageReward med_config.json --local-dir ./reward_ckpt/image_reward
```

Configuration example:

```yaml
reward:
    imagereward_model_name: "./reward_ckpt/image_reward/ImageReward.pt"
    imagereward_med_config: "./reward_ckpt/image_reward/med_config.json"
```

---

## Step 10.3: GenEval HTTP Service

The `geneval` reward in TMPO uses HTTP client mode, not a local function.

You need to start a compatible reward server first and configure:

```yaml
reward:
    geneval_url: ""
    geneval_batch_size: 64
    geneval_timeout: 120
    geneval_only_strict: true
```

---

## Step 11: Prepare Prompt Data

```bash
mkdir -p data
```

**Option A**: Download from Pick-a-Pic (recommended)
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('yuvalkirstain/pickapic_v2', split='train')
prompts = list(set(ds['caption']))[:2000]
import json
with open('data/pickapic_prompts.json', 'w') as f:
    json.dump(prompts, f, indent=2)
print(f'Saved {len(prompts)} prompts')
"
```

**Option B**: Create sample data manually
```bash
python -c "
import json
prompts = [
    'a photo of a cat sitting on a windowsill',
    'an oil painting of mountains at sunset',
    'a digital illustration of a futuristic city',
] * 100
with open('data/pickapic_prompts.json', 'w') as f:
    json.dump(prompts, f, indent=2)
print(f'Created {len(prompts)} sample prompts')
"
```

**Verify**: `python -c "import json; print(len(json.load(open('data/pickapic_prompts.json'))))"`

---

## Step 12: Create Output Directories

```bash
mkdir -p outputs/sd35m outputs/sd35m_fullparam
```

---

## Step 13: Full Environment Verification

```bash
python -c "
import torch, accelerate, diffusers, peft, open_clip, yaml, os

print('=== Environment Check ===')
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA:        {torch.version.cuda}')
print(f'GPUs:        {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name} ({mem:.0f} GB)')
print(f'Accelerate:  {accelerate.__version__}')
print(f'Diffusers:   {diffusers.__version__}')
print(f'PEFT:        {peft.__version__}')

print()
print('=== File Check ===')
checks = {
    'HPSv2 weights': 'reward_ckpt/HPS_v2.1_compressed.pt',
    'OpenCLIP weights': 'reward_ckpt/open_clip_pytorch_model.bin',
    'Prompt data': 'data/pickapic_prompts.json',
    'LoRA training config': 'config/sd35m_lora.yaml',
    'FSDP config': 'accelerate_configs/fsdp_small.yaml',
}
for name, path in checks.items():
    exists = os.path.exists(path)
    print(f'  {\"OK\" if exists else \"MISSING\"} {name}: {path}')
"
```

---

## Step 14: Start Training

### LoRA Training (recommended to try first)

```bash
bash scripts/train_flux.sh
```

### Manual Launch (override parameters)

```bash
accelerate launch \
    --config_file accelerate_configs/fsdp_small.yaml \
    --num_processes 4 \
    tmpo/train.py --config config/sd35m_lora.yaml --lr 1e-5
```

---

## Step 15: Monitor Training

### Terminal Logs
```bash
tail -f outputs/sd35m/training.log
```

### Wandb Dashboard
The terminal will output a wandb link:
```
wandb: View run at https://wandb.ai/your-team/tmpo/runs/xxxxx
```
Click the link to view real-time curves in your browser.

---

## LoRA vs Full-Parameter Comparison

| | LoRA (r=32) | Full Params |
|---|---|---|
| **Config** | `config/sd35m_lora.yaml` | - |
| **Trainable params** | ~2% (~50M) | 100% (~2.5B) |
| **VRAM/GPU** | ~15GB | ~55GB |
| **Learning rate** | 1e-5 | 3e-6 |
| **weight_decay** | 1e-4 | 0.01 |
| **Convergence** | Slower | Faster |
| **Overfitting risk** | Low | High (needs stronger regularization) |
| **Checkpoint size** | ~200MB (LoRA weights only) | ~10GB (full model) |
