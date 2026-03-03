"""检查点管理

支持 LoRA 权重和优化器状态的保存/加载。
"""

import os
import json
import torch
from typing import Optional


def save_checkpoint(
    accelerator,
    transformer,
    optimizer,
    step: int,
    epoch: int,
    output_dir: str,
    is_lora: bool = True,
    pipeline=None,
):
    """保存训练检查点

    Args:
        accelerator: Accelerator 实例
        transformer: 模型 (可能被 FSDP 包装)
        optimizer: 优化器
        step: 当前步数
        epoch: 当前 epoch
        output_dir: 输出目录
        is_lora: 是否为 LoRA 模型
        pipeline: diffusers Pipeline (LoRA 保存需要)
    """
    if not accelerator.is_main_process:
        return

    save_dir = os.path.join(output_dir, f"checkpoint-{step}-{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    if is_lora and pipeline is not None:
        # LoRA 权重保存
        try:
            from peft import get_peft_model_state_dict
            unwrapped = accelerator.unwrap_model(transformer)
            lora_state = get_peft_model_state_dict(unwrapped)
            pipeline.save_lora_weights(
                save_directory=save_dir,
                transformer_lora_layers=lora_state,
                is_main_process=True,
            )
        except ImportError:
            # Fallback: 直接保存 state_dict
            unwrapped = accelerator.unwrap_model(transformer)
            torch.save(unwrapped.state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        # 全参数保存
        unwrapped = accelerator.unwrap_model(transformer)
        torch.save(unwrapped.state_dict(), os.path.join(save_dir, "model.pt"))

    # 保存优化器状态
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    # 保存训练元信息
    meta = {"step": step, "epoch": epoch}
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Checkpoint] Saved at step {step} to {save_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    transformer=None,
    optimizer=None,
):
    """加载检查点

    Returns:
        step: 恢复的步数
    """
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    model_path = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(model_path) and transformer is not None:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        transformer.load_state_dict(state_dict, strict=False)

    optim_path = os.path.join(checkpoint_dir, "optimizer.pt")
    if os.path.exists(optim_path) and optimizer is not None:
        optim_state = torch.load(optim_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(optim_state)

    print(f"[Checkpoint] Loaded from {checkpoint_dir}, step={meta['step']}")
    return meta["step"]
