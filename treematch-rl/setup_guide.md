# TreeMatch-RL 8×H200 逐步部署指南

> 不是一个大脚本, 是一步一步来。每步执行完确认再下一步。

---

## Step 1: 创建 conda 环境

```bash
conda create -n treematch python=3.10 -y
conda activate treematch
```

**验证**: `python --version` 应显示 `3.10.x`

---

## Step 2: 安装 PyTorch

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

**验证**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs {torch.cuda.device_count()}')"
```
应输出: `PyTorch 2.3.0, CUDA 12.1, GPUs 8`

---

## Step 3: 安装训练依赖

```bash
pip install accelerate==0.33.0
pip install diffusers==0.30.0
pip install transformers==4.44.0
pip install peft
```

**验证**:
```bash
python -c "import accelerate, diffusers; print(f'Accelerate {accelerate.__version__}, Diffusers {diffusers.__version__}')"
```

---

## Step 4: 安装奖励模型依赖

```bash
pip install open_clip_torch
pip install pyyaml Pillow numpy
```

---

## Step 5: 安装 Wandb

```bash
pip install wandb
wandb login
# 输入你的 API key (从 https://wandb.ai/authorize 获取)
```

---

## Step 6: 登录 HuggingFace

SD3.5-Medium 需要接受 License Agreement。

```bash
pip install huggingface_hub
huggingface-cli login
# 输入你的 HF token (从 https://huggingface.co/settings/tokens 获取)
```

然后在浏览器访问 https://huggingface.co/stabilityai/stable-diffusion-3.5-medium 点击 **Agree and access**。

**验证**:
```bash
python -c "from huggingface_hub import HfFolder; print('Token:', 'OK' if HfFolder.get_token() else 'MISSING')"
```

---

## Step 7: 下载 SD3.5-Medium 模型

模型约 10GB, 直接下载到本地 `models/` 目录:

```bash
mkdir -p models/sd35m

huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir ./models/sd35m
```

> ⚠️ 如果报权限错误, 请先在浏览器访问 https://huggingface.co/stabilityai/stable-diffusion-3.5-medium 点击 **Agree and access**

**验证**: `ls models/sd35m/transformer/`

---

## Step 8: 下载 HPSv2 奖励模型权重

```bash
mkdir -p reward_ckpt

# HPSv2 主权重
hf download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./reward_ckpt/
```

**验证**: `ls -la reward_ckpt/HPS_v2.1_compressed.pt`

---

## Step 9: 下载 OpenCLIP 权重 (HPSv2 的 backbone)

```bash
hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./reward_ckpt/
```

**验证**: `ls -la reward_ckpt/open_clip_pytorch_model.bin`

---

## Step 10: CLIP Score 模型

```bash
mkdir -p reward_ckpt/clip_score

hf download apple/DFN5B-CLIP-ViT-H-14-384 --local-dir ./reward_ckpt/clip_score
```

**验证**: `ls reward_ckpt/clip_score/open_clip_pytorch_model.bin`

---

## Step 11: 准备 Prompt 数据

```bash
mkdir -p data
```

**方式 A**: 从 Pick-a-Pic 下载 (推荐)
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

**方式 B**: 手动创建示例数据
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

**验证**: `python -c "import json; print(len(json.load(open('data/pickapic_prompts.json'))))"`

---

## Step 12: 创建输出目录

```bash
mkdir -p outputs/sd35m outputs/sd35m_fullparam
```

---

## Step 13: 全环境验证

```bash
python -c "
import torch, accelerate, diffusers, peft, open_clip, yaml, os

print('=== 环境检查 ===')
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
print('=== 文件检查 ===')
checks = {
    'HPSv2 权重': 'reward_ckpt/HPS_v2.1_compressed.pt',
    'OpenCLIP 权重': 'reward_ckpt/open_clip_pytorch_model.bin',
    'Prompt 数据': 'data/pickapic_prompts.json',
    'LoRA 训练配置': 'config/sd35m_lora.yaml',
    '全参数训练配置': 'config/sd35m_fullparam.yaml',
    'FSDP LoRA 配置': 'accelerate_configs/fsdp_8gpu.yaml',
    'FSDP 全参数配置': 'accelerate_configs/fsdp_8gpu_fullparam.yaml',
}
for name, path in checks.items():
    exists = os.path.exists(path)
    print(f'  {\"✅\" if exists else \"❌\"} {name}: {path}')
"
```

---

## Step 14: 启动训练

### LoRA 训练 (推荐先试)

```bash
bash scripts/train_8xH200.sh
```

### 全参数训练

```bash
bash scripts/train_8xH200_fullparam.sh
```

### 手动启动 (可覆盖参数)

```bash
# LoRA
accelerate launch \
    --config_file accelerate_configs/fsdp_8gpu.yaml \
    --num_processes 8 \
    treematch/train.py --config config/sd35m_lora.yaml --lr 1e-5

# 全参数
accelerate launch \
    --config_file accelerate_configs/fsdp_8gpu_fullparam.yaml \
    --num_processes 8 \
    treematch/train.py --config config/sd35m_fullparam.yaml --lr 3e-6
```

---

## Step 15: 查看训练

### Terminal 日志
```bash
# 实时查看
tail -f outputs/sd35m/training.log

# 或 outputs/sd35m_fullparam/training.log
```

### Wandb 面板
终端会输出 wandb 链接:
```
wandb: 🚀 View run at https://wandb.ai/your-team/treematch-rl/runs/xxxxx
```
点击链接在浏览器查看实时曲线。

---

## LoRA vs 全参数 对比

| | LoRA (r=32) | 全参数 |
|---|---|---|
| **配置文件** | `config/sd35m_lora.yaml` | `config/sd35m_fullparam.yaml` |
| **Accelerate** | `fsdp_8gpu.yaml` (SHARD_GRAD_OP) | `fsdp_8gpu_fullparam.yaml` (FULL_SHARD) |
| **可训练参数** | ~2% (~50M) | 100% (~2.5B) |
| **显存/卡** | ~15GB | ~55GB |
| **学习率** | 1e-5 | 3e-6 |
| **weight_decay** | 1e-4 | 0.01 |
| **收敛速度** | 较慢 | 较快 |
| **过拟合风险** | 低 | 高 → 需要更强正则 |
| **检查点大小** | ~200MB (仅 LoRA 权重) | ~10GB (全模型) |
