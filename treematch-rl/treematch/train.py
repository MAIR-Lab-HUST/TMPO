"""TreeMatch-RL 主训练入口

完整训练循环:
    1. 加载模型 (SD3.5/Flux + LoRA + VAE)
    2. 自适应调度: Beta 分布 → 分叉位置 + 噪声系数
    3. 树状采样: 3 阶 27 分支 + DPM Flash 加速
    4. VAE 解码 + 多奖励评分
    5. Softmax-TB 损失 + IS 多次更新
    6. FSDP 分布式参数更新
"""

import os
import json
import argparse
import math
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from treematch.sampling import TreeSampler, AdaptiveScheduler
from treematch.sampling.scheduler import build_sigma_schedule
from treematch.losses import TreeMatchRLLoss
from treematch.losses.entropy import ParticleEntropyLoss
from treematch.rewards.compute import build_reward_models, decode_and_compute_rewards
from treematch.utils.logging_ import setup_logging, main_print
from treematch.utils.checkpoint import save_checkpoint
from treematch.utils.distributed import gather_rewards


# ═══════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════
class PromptDataset(Dataset):
    """Prompt 数据集"""

    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # 支持 list 或 dict 格式
        if isinstance(self.data, dict):
            self.prompts = list(self.data.values())
        elif isinstance(self.data, list):
            self.prompts = [
                item["prompt"] if isinstance(item, dict) else item
                for item in self.data
            ]
        else:
            self.prompts = [str(self.data)]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


# ═══════════════════════════════════════════════════
# 训练一步
# ═══════════════════════════════════════════════════
def train_one_step(
    accelerator,
    transformer,
    vae,
    pipeline,
    tree_sampler,
    scheduler,
    loss_fn,
    optimizer,
    lr_scheduler,
    prompt: str,
    reward_models,
    reward_weights,
    config,
    step: int,
    prev_mean_reward: float = None,
):
    """执行完整的一步训练

    Returns:
        metrics: 训练指标字典
        mean_reward: 本步平均奖励 (用于下一步自适应调度)
    """
    device = accelerator.device
    dtype = torch.bfloat16

    tree_cfg = config["tree"]
    loss_cfg = config["loss"]
    train_cfg = config["training"]

    # ═══ 1. 自适应调度: 决定分叉位置和噪声系数 ═══
    split_steps, noise_levels, alpha = scheduler.get_schedule(prev_mean_reward)
    main_print(
        f"[Step {step}] alpha={alpha:.2f}, "
        f"splits={split_steps}, noise={[f'{n:.2f}' for n in noise_levels]}"
    )

    # ═══ 2. 文本编码 ═══
    with torch.no_grad():
        text_inputs = pipeline.tokenizer(
            [prompt],
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # 获取文本嵌入 (SD3 需要多个编码器)
        prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        if isinstance(prompt_embeds, tuple):
            encoder_hidden_states = prompt_embeds[0]
            pooled_prompt_embeds = prompt_embeds[1] if len(prompt_embeds) > 1 else None
        else:
            encoder_hidden_states = prompt_embeds
            pooled_prompt_embeds = None

    # Latent 尺寸
    H, W = config["dataset"]["resolution"]
    latent_h, latent_w = H // 8, W // 8
    latent_channels = 16  # SD3 latent channels
    latent_shape = (1, latent_channels, latent_h, latent_w)

    # Dummy ids (SD3 需要)
    text_ids = torch.zeros(encoder_hidden_states.shape[1], 3, device=device)
    latent_image_ids = torch.zeros(latent_h * latent_w, 3, device=device)

    # ═══ 3. 树状采样 (no_grad) ═══
    branches = tree_sampler.sample(
        transformer=accelerator.unwrap_model(transformer),
        latent_shape=latent_shape,
        encoder_hidden_states=encoder_hidden_states,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        latent_image_ids=latent_image_ids,
        split_steps=split_steps,
        noise_levels=noise_levels,
        device=device,
        dtype=dtype,
    )

    K = len(branches)
    main_print(f"[Step {step}] Sampled {K} branches")

    # ═══ 4. VAE 解码 + 奖励计算 ═══
    all_latents = torch.stack([b["latent"] for b in branches]).to(device)
    prompts_list = [prompt] * K

    rewards, rewards_dict = decode_and_compute_rewards(
        latents=all_latents,
        vae=vae,
        prompts=prompts_list,
        reward_models=reward_models,
        reward_weights=reward_weights,
        mix_strategy=config["reward"].get("mix_strategy", "advantage_aggr"),
        batch_size=train_cfg.get("vae_decode_batch_size", 4),
    )
    rewards = rewards.to(device)

    # 更新调度器的奖励边界
    scheduler.update_reward_bounds(rewards)
    mean_reward = rewards.mean().item()

    # ═══ 5. 旧策略 log_prob (detach) ═══
    old_log_probs = torch.tensor(
        [b["log_prob_sum"] for b in branches],
        device=device, dtype=torch.float32,
    )

    # ═══ 6. 参考模型 log_prob (初始时与旧策略相同) ═══
    ref_log_probs = old_log_probs.clone().detach()

    # ═══ 7. 路径特征 (for 粒子熵) ═══
    path_features = ParticleEntropyLoss.compute_latent_features(all_latents.float())

    # ═══ 8. IS 多次更新 ═══
    all_metrics = []
    sigmas = build_sigma_schedule(
        tree_cfg["num_inference_steps"], tree_cfg.get("shift", 3.0), device
    )

    for update_iter in range(loss_cfg.get("is_num_updates", 4)):
        # 重新计算当前策略的 log_prob (需要梯度) + RatioNorm 逐步数据
        recompute_result = tree_sampler.recompute_path_log_probs(
            transformer=transformer,
            branches=branches,
            split_steps=split_steps,
            noise_levels=noise_levels,
            sigmas=sigmas,
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            dtype=dtype,
        )

        current_log_probs = recompute_result["path_log_probs"]

        # 计算总损失 (传入 RatioNorm 逐步数据)
        loss, metrics = loss_fn(
            current_log_probs=current_log_probs,
            old_log_probs=old_log_probs,
            rewards=rewards,
            ref_log_probs=ref_log_probs,
            path_features=path_features.detach(),
            num_sde_steps=len(split_steps),
            # RatioNorm 逐步数据
            step_log_probs=recompute_result["step_log_probs"],
            old_step_log_probs=recompute_result["old_step_log_probs"],
            step_means=recompute_result["step_means"],
            old_step_means=recompute_result["old_step_means"],
            std_dev_ts=recompute_result["std_dev_ts"],
            sqrt_dts=recompute_result["sqrt_dts"],
        )

        # 梯度累积
        grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
        scaled_loss = loss / grad_accum

        accelerator.backward(scaled_loss)

        if (update_iter + 1) % grad_accum == 0 or update_iter == loss_cfg.get("is_num_updates", 4) - 1:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    transformer.parameters(), train_cfg.get("max_grad_norm", 1.0)
                )
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

        # 更新 old_log_probs (用于下一轮 IS)
        old_log_probs = current_log_probs.detach()

        metrics["update_iter"] = update_iter
        metrics["mean_reward"] = mean_reward
        metrics["alpha"] = alpha
        all_metrics.append(metrics)

    return all_metrics[-1], mean_reward


# ═══════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="TreeMatch-RL Training")
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    tree_cfg = config["tree"]
    loss_cfg = config["loss"]

    # ═══ Accelerator ═══
    accelerator = Accelerator(
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
    )

    set_seed(train_cfg.get("seed", 42))

    # 日志
    logger = setup_logging(
        log_dir=train_cfg.get("output_dir"),
        rank=accelerator.process_index,
    )
    main_print("=" * 60)
    main_print("TreeMatch-RL Training")
    main_print("=" * 60)
    main_print(f"Config: {args.config}")
    main_print(f"Num GPUs: {accelerator.num_processes}")

    if train_cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ═══ 加载模型 ═══
    model_cfg = config["model"]
    main_print(f"Loading model: {model_cfg['pretrained_path']}")

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_cfg["pretrained_path"],
        torch_dtype=torch.bfloat16,
    )

    transformer = pipeline.transformer
    vae = pipeline.vae
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    # ═══ LoRA ═══
    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg:
        main_print(f"Applying LoRA: rank={lora_cfg.get('rank', 32)}")
        lora_config = LoraConfig(
            r=lora_cfg.get("rank", 32),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
            lora_dropout=lora_cfg.get("dropout", 0.0),
            bias="none",
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()
    else:
        transformer.requires_grad_(True)

    # 梯度检查点
    if train_cfg.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()

    # ═══ 优化器 ═══
    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=float(train_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("max_train_steps", 300),
        eta_min=float(train_cfg.get("learning_rate", 1e-5)) * 0.1,
    )

    # ═══ 数据集 ═══
    dataset = PromptDataset(config["dataset"]["data_json_path"])
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
    )
    main_print(f"Dataset: {len(dataset)} prompts")

    # ═══ Accelerator.prepare ═══
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler,
    )

    # ═══ 奖励模型 ═══
    reward_models, reward_weights = build_reward_models(
        config["reward"], accelerator.device,
    )
    main_print(f"Reward models: {list(reward_models.keys())}")

    # ═══ 采样器 & 调度器 & 损失 ═══
    tree_sampler = TreeSampler(
        num_inference_steps=tree_cfg.get("num_inference_steps", 28),
        k=tree_cfg.get("k", 3),
        shift=tree_cfg.get("shift", 3.0),
        dpm_flash_enabled=config.get("dpm_flash", {}).get("enabled", True),
        dpm_compress_ratio=config.get("dpm_flash", {}).get("compress_ratio", 0.4),
        dpm_order=config.get("dpm_flash", {}).get("solver_order", 2),
        dpm_solver_type=config.get("dpm_flash", {}).get("solver_type", "midpoint"),
    )

    scheduler = AdaptiveScheduler(
        num_inference_steps=tree_cfg.get("num_inference_steps", 28),
        num_splits=3,
        kappa=tree_cfg.get("kappa", 4.0),
        base_noise_levels=tree_cfg.get("base_noise_levels", [0.4, 0.7, 1.0]),
    )

    loss_fn = TreeMatchRLLoss(
        beta=loss_cfg.get("beta", 15.0),
        lambda_entropy=loss_cfg.get("lambda_entropy", 0.01),
        lambda_ref=loss_cfg.get("lambda_ref", 0.1),
        is_clip_range=loss_cfg.get("is_clip_range", 0.2),
        rbf_bandwidth=loss_cfg.get("rbf_bandwidth", 1.0),
    )

    # ═══ 训练循环 ═══
    max_steps = train_cfg.get("max_train_steps", 300)
    checkpoint_steps = train_cfg.get("checkpointing_steps", 50)
    output_dir = train_cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    main_print(f"Starting training: {max_steps} steps")
    main_print("-" * 60)

    global_step = 0
    prev_mean_reward = None
    data_iter = iter(dataloader)

    while global_step < max_steps:
        # 取下一个 prompt
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]

        # 训练一步
        metrics, prev_mean_reward = train_one_step(
            accelerator=accelerator,
            transformer=transformer,
            vae=vae,
            pipeline=pipeline,
            tree_sampler=tree_sampler,
            scheduler=scheduler,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            prompt=prompt,
            reward_models=reward_models,
            reward_weights=reward_weights,
            config=config,
            step=global_step,
            prev_mean_reward=prev_mean_reward,
        )

        global_step += 1

        # 日志
        if global_step % 1 == 0:
            main_print(
                f"[Step {global_step}/{max_steps}] "
                f"loss={metrics['loss_total']:.4f} "
                f"tb={metrics['loss_soft_tb']:.4f} "
                f"entropy={metrics['loss_entropy']:.4f} "
                f"ref={metrics['loss_ref']:.4f} "
                f"reward={metrics['mean_reward']:.4f} "
                f"α={metrics.get('alpha', 0):.2f}"
            )

        # 检查点
        if global_step % checkpoint_steps == 0:
            save_checkpoint(
                accelerator=accelerator,
                transformer=transformer,
                optimizer=optimizer,
                step=global_step,
                epoch=0,
                output_dir=output_dir,
                is_lora=bool(lora_cfg),
                pipeline=pipeline,
            )

    # 最终保存
    save_checkpoint(
        accelerator=accelerator,
        transformer=transformer,
        optimizer=optimizer,
        step=global_step,
        epoch=0,
        output_dir=output_dir,
        is_lora=bool(lora_cfg),
        pipeline=pipeline,
    )

    main_print("=" * 60)
    main_print("Training complete!")
    main_print("=" * 60)


if __name__ == "__main__":
    main()
