"""检查点管理

支持 LoRA 权重和优化器状态的保存/加载。
"""

import os
import json
import torch
from typing import Optional, Dict


def _extract_local_lora_state(module) -> Dict[str, torch.Tensor]:
    """提取当前 rank 的 LoRA 可训练参数快照（不触发 full state_dict 汇总）。"""
    local_state: Dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if "lora" in lname:
            local_state[name] = param.detach().cpu().clone()

    # 兜底: 若未命中 lora 命名, 至少保存所有 requires_grad 参数
    if not local_state:
        for name, param in module.named_parameters():
            if param.requires_grad:
                local_state[name] = param.detach().cpu().clone()

    return local_state


def save_checkpoint(
    accelerator,
    transformer,
    optimizer,
    step: int,
    epoch: int,
    output_dir: str,
    is_lora: bool = True,
    pipeline=None,
    save_optimizer: bool = False,
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
    save_dir = os.path.join(output_dir, f"checkpoint-{step}-{epoch}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    rank = int(getattr(accelerator, "process_index", 0))
    world_size = int(getattr(accelerator, "num_processes", 1))
    unwrapped = accelerator.unwrap_model(transformer)

    if is_lora:
        # 分布式下每个 rank 各自保存本地 LoRA shard, 避免 checkpoint 时全量通信。
        local_lora_state = _extract_local_lora_state(unwrapped)
        shard_path = os.path.join(save_dir, f"lora_rank{rank:02d}.pt")
        torch.save(local_lora_state, shard_path)

        # 单卡时额外导出 diffusers 兼容格式, 便于直接推理加载。
        if world_size == 1 and pipeline is not None:
            try:
                from peft import get_peft_model_state_dict
                lora_state = get_peft_model_state_dict(unwrapped)
                pipeline.save_lora_weights(
                    save_directory=save_dir,
                    transformer_lora_layers=lora_state,
                    is_main_process=True,
                )
            except Exception:
                pass
    else:
        # 全参数模式: 保持主进程保存行为
        if accelerator.is_main_process:
            torch.save(unwrapped.state_dict(), os.path.join(save_dir, "model.pt"))

    # 优化器状态默认关闭(LoRA 场景体积大且易导致 checkpoint 卡顿)
    if save_optimizer:
        optim_path = os.path.join(save_dir, f"optimizer_rank{rank:02d}.pt")
        torch.save(optimizer.state_dict(), optim_path)

    if accelerator.is_main_process:
        meta = {
            "step": step,
            "epoch": epoch,
            "is_lora": bool(is_lora),
            "world_size": world_size,
            "format": "lora_local_shards" if is_lora else "full_model",
            "save_optimizer": bool(save_optimizer),
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Checkpoint] Saved at step {step} to {save_dir}")

    accelerator.wait_for_everyone()


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
