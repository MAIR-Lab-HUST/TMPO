"""Checkpoint management for LoRA weights and optimizer state."""

import json
import os
from typing import Dict

import torch


def _extract_local_lora_state(module) -> Dict[str, torch.Tensor]:
    """Extract local LoRA trainable params snapshot (no full state_dict gather)."""
    local_state: Dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if "lora" in lname:
            local_state[name] = param.detach().cpu().clone()

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
    """Save training checkpoint."""
    save_dir = os.path.join(output_dir, f"checkpoint-{step}-{epoch}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    rank = int(getattr(accelerator, "process_index", 0))
    world_size = int(getattr(accelerator, "num_processes", 1))
    unwrapped = accelerator.unwrap_model(transformer)

    if is_lora:
        local_lora_state = _extract_local_lora_state(unwrapped)
        shard_path = os.path.join(save_dir, f"lora_rank{rank:02d}.pt")
        torch.save(local_lora_state, shard_path)
    else:
        if accelerator.is_main_process:
            torch.save(unwrapped.state_dict(), os.path.join(save_dir, "model.pt"))

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
    """Load checkpoint and return the restored step."""
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
