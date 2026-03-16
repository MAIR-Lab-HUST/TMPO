#!/bin/bash
# ═══════════════════════════════════════════════════════════
# TreeMatch-RL 服务器环境一键配置脚本
# 用法: bash scripts/setup_server.sh
# ═══════════════════════════════════════════════════════════

set -e

echo "════════════════════════════════════════"
echo "  TreeMatch-RL 服务器环境配置"
echo "════════════════════════════════════════"

# ─── 0. 检查基本环境 ───
echo ""
echo "[0/6] 检查基本环境..."
if ! command -v conda &> /dev/null; then
    echo "⚠️  conda 未安装。请先安装 Miniconda:"
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "    bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi
echo "✅ conda: $(conda --version)"

if ! nvidia-smi &> /dev/null; then
    echo "⚠️  nvidia-smi 不可用, 请确认 GPU 驱动已安装"
    exit 1
fi
echo "✅ GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ─── 1. 创建 conda 环境 ───
echo ""
echo "[1/6] 创建 conda 环境 (treematch, Python 3.10)..."
if conda info --envs | grep -q "treematch"; then
    echo "  环境已存在, 跳过创建"
else
    conda create -n treematch python=3.10 -y
fi

# 激活环境 (在脚本中通过 conda run 执行)
CONDA_RUN="conda run -n treematch --no-capture-output"

# ─── 2. 安装 Python 依赖 ───
echo ""
echo "[2/6] 安装 Python 依赖..."

$CONDA_RUN pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    2>&1 | tail -1

$CONDA_RUN pip install \
    accelerate==0.33.0 \
    diffusers==0.30.0 \
    transformers==4.44.0 \
    peft \
    open_clip_torch \
    pyyaml \
    Pillow \
    numpy \
    wandb \
    2>&1 | tail -1

echo "✅ Python 依赖安装完成"

# ─── 3. 下载基座模型 ───
echo ""
echo "[3/6] 下载基座模型 (SD3.5-Medium)..."
echo "  模型将通过 HuggingFace 在首次运行时自动下载"
echo "  如果需要提前下载, 运行:"
echo "    $CONDA_RUN python -c \"from diffusers import StableDiffusion3Pipeline; StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3.5-medium', torch_dtype='auto')\""
echo ""
echo "  ⚠️  注意: SD3.5-Medium 需要接受 HuggingFace License Agreement"
echo "  请先访问: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
echo "  然后运行: huggingface-cli login"

# 检查 HuggingFace token
if $CONDA_RUN python -c "from huggingface_hub import HfFolder; t=HfFolder.get_token(); assert t" 2>/dev/null; then
    echo "✅ HuggingFace token 已配置"
else
    echo "⚠️  HuggingFace token 未配置, 请运行: huggingface-cli login"
fi

# ─── 4. 下载奖励模型权重 ───
echo ""
echo "[4/6] 下载奖励模型权重..."

REWARD_DIR="reward_ckpt"
mkdir -p "$REWARD_DIR"

# 4a. HPSv2 权重
HPS_WEIGHT="$REWARD_DIR/HPS_v2.1_compressed.pt"
if [ -f "$HPS_WEIGHT" ]; then
    echo "  ✅ HPSv2 权重已存在: $HPS_WEIGHT"
else
    echo "  ⬇️  下载 HPSv2 权重..."
    $CONDA_RUN python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='xswu/HPSv2',
    filename='HPS_v2.1_compressed.pt',
    local_dir='$REWARD_DIR',
)
print(f'  ✅ HPSv2 下载完成: {path}')
" || echo "  ⚠️  HPSv2 下载失败, 请手动下载: https://huggingface.co/xswu/HPSv2"
fi

# 4b. OpenCLIP 权重 (HPSv2 的 CLIP backbone)
CLIP_WEIGHT="$REWARD_DIR/open_clip_pytorch_model.bin"
if [ -f "$CLIP_WEIGHT" ]; then
    echo "  ✅ OpenCLIP 权重已存在: $CLIP_WEIGHT"
else
    echo "  ⬇️  下载 OpenCLIP 权重 (ViT-H-14)..."
    $CONDA_RUN python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    filename='open_clip_pytorch_model.bin',
    local_dir='$REWARD_DIR',
)
print(f'  ✅ OpenCLIP 下载完成: {path}')
" || echo "  ⚠️  OpenCLIP 下载失败, 请手动下载"
fi

# 4c. CLIP Score 模型 (DFN5B, 自动下载)
echo "  ℹ️  CLIP Score (apple/DFN5B-CLIP-ViT-H-14-384) 将在首次运行时自动下载"
echo "     配置路径: hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"

# ─── 5. 准备数据 ───
echo ""
echo "[5/6] 准备 Prompt 数据..."

DATA_DIR="data"
mkdir -p "$DATA_DIR"

PROMPT_FILE="$DATA_DIR/pickapic_prompts.json"
if [ -f "$PROMPT_FILE" ]; then
    echo "  ✅ Prompt 数据已存在: $PROMPT_FILE"
    PROMPT_COUNT=$($CONDA_RUN python -c "import json; print(len(json.load(open('$PROMPT_FILE'))))" 2>/dev/null || echo "?")
    echo "     包含 $PROMPT_COUNT 条 prompt"
else
    echo "  ⬇️  生成示例 Prompt 数据 (500 条)..."
    $CONDA_RUN python -c "
import json

# 示例 prompt 集 (实际使用时请替换为 Pick-a-Pic 数据)
prompts = [
    'a professional photograph of a golden retriever playing in autumn leaves',
    'an oil painting of a sunset over a mountain lake',
    'a digital art illustration of a futuristic city at night',
    'a watercolor painting of cherry blossoms in spring',
    'a macro photograph of a dewdrop on a spider web',
    'a fantasy illustration of a dragon flying over a medieval castle',
    'a minimalist photograph of geometric architecture',
    'a portrait painting in the style of Rembrandt',
    'a surreal digital art of floating islands in the sky',
    'a photograph of a cozy cafe interior with warm lighting',
] * 50  # 重复到 500 条

with open('$PROMPT_FILE', 'w') as f:
    json.dump(prompts, f, indent=2)
print(f'  ✅ 已生成 {len(prompts)} 条示例 prompt')
print('  ⚠️  实际训练请替换为 Pick-a-Pic prompt 数据:')
print('     https://huggingface.co/datasets/yuvalkirstain/pickapic_v2')
"
fi

# ─── 6. 创建输出目录 ───
echo ""
echo "[6/6] 创建输出目录..."
mkdir -p outputs/sd35m
mkdir -p outputs/flux
echo "✅ 输出目录: outputs/sd35m/, outputs/flux/"

# ─── 验证 ───
echo ""
echo "════════════════════════════════════════"
echo "  环境验证"
echo "════════════════════════════════════════"
$CONDA_RUN python -c "
import torch
import accelerate
import diffusers
import peft
import open_clip
import yaml

print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA:        {torch.version.cuda}')
print(f'  GPUs:        {torch.cuda.device_count()}')
print(f'  Accelerate:  {accelerate.__version__}')
print(f'  Diffusers:   {diffusers.__version__}')
print(f'  PEFT:        {peft.__version__}')

import os
hps = os.path.exists('reward_ckpt/HPS_v2.1_compressed.pt')
clip = os.path.exists('reward_ckpt/open_clip_pytorch_model.bin')
data = os.path.exists('data/pickapic_prompts.json')
print(f'  HPSv2:       {\"✅\" if hps else \"❌ 缺失\"}')
print(f'  OpenCLIP:    {\"✅\" if clip else \"❌ 缺失\"}')
print(f'  Prompt Data: {\"✅\" if data else \"❌ 缺失\"}')
"

echo ""
echo "════════════════════════════════════════"
echo "  配置完成! 下一步:"
echo "════════════════════════════════════════"
echo ""
echo "  1. 激活环境:  conda activate treematch"
echo "  2. 登录 wandb: wandb login"
echo "  3. 登录 HF:   huggingface-cli login"
echo "  4. 开始训练:  bash scripts/train_8xH200.sh"
echo ""
