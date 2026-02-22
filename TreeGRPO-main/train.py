import datetime
import os
import random

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from collections import defaultdict
from diffusers import StableDiffusion3Pipeline
from omegaconf import DictConfig
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TextPromptDataset
from pipelines_with_logprob.stablediffusion3 import sd3_pipeline_with_logprob
from reward_models.hps import HPS_v2
from schedulers_with_logprob.flow_match_euler_discrete import flow_match_euler_discrete_step_with_logprob

class RLTrainer:
    """Reinforcement Learning trainer for diffusion models using TreeGRPO."""
    
    def __init__(self, config, accelerator, log_path='rl_training', seed=42):
        self.config = config
        self.accelerator = accelerator
        self.global_step = 0
        self.epoch_count = 0
        self.tree_steps = []
        if self.config.tree.enable:
            self.tree_steps = [i for i in range(self.config.tree.w)]
        set_seed(seed, device_specific=True)
        
        if accelerator.is_main_process:
            os.makedirs(config.training.ckpt_dir, exist_ok=True)
        
        self.accelerator.init_trackers(log_path)
        self.inference_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            self.inference_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            self.inference_dtype = torch.bfloat16

        if self.config.env.pipeline == "StableDiffusion3":
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                self.config.env.model_name
            )
            self.denoiser = self.pipe.transformer
            self.pipe.text_encoder.requires_grad_(False)
            self.pipe.text_encoder_2.requires_grad_(False)
            self.pipe.text_encoder_3.requires_grad_(False)
            self.pipe.vae.requires_grad_(False)
            self.pipe.transformer.enable_gradient_checkpointing()
            self.pipe.transformer.requires_grad_(True)

            self.pipe.text_encoder.to(dtype=self.inference_dtype)
            self.pipe.text_encoder_2.to(dtype=self.inference_dtype)
            self.pipe.text_encoder_3.to(dtype=self.inference_dtype)
            self.pipe.vae.to(dtype=self.inference_dtype)
            self.pipe.vae.to(self.accelerator.device)
        else:
            raise NotImplementedError(f"pipeline: {self.config.env.pipeline} not supported.")

        self.pipe.set_progress_bar_config(
            position=1,
            disable=not accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # Load reward model
        if self.config.env.reward_model == "hpsv2":
            self.accelerator.print("Loading HPSv2 reward model...")
            self.reward_model = HPS_v2(device=self.accelerator.device)
        elif self.config.env.reward_model == "none":
            # use dummy reward model, for debug only
            self.reward_model = None
        else:
            raise NotImplementedError(f"Reward model '{self.config.env.reward_model}' not supported.")

        self.optimizer = AdamW(
            self.denoiser.parameters(),
            lr=config.training.lr,
            weight_decay=1e-4,
        )
        if self.config.env.pipeline == "StableDiffusion3":
            self.denoiser, self.optimizer, self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.text_encoder_3 = self.accelerator.prepare(
                self.denoiser, self.optimizer, self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.text_encoder_3
            )
        else:
            raise NotImplementedError(f"pipeline: {self.config.env.pipeline} not supported.")

        self._create_dataloader()
    
    def _create_dataloader(self):
        dataset = TextPromptDataset(self.config.data.prompt_path)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.sample.num_prompts,
            shuffle=True,
            num_workers=1,
        )
        self.dataloader = self.accelerator.prepare(dataloader)

    def calculate_rewards(self, root_node, prompt):
        """Calculate rewards for all leaf images in the tree."""
        nodes = [root_node]
        all_rewards = []
        while nodes:
            node = nodes.pop(0)
            if node.image:
                if self.reward_model:
                    score = self.reward_model(node.image, prompt)
                else:
                    score = torch.tensor([0.]).to(self.accelerator.device)
                node.reward = score
                all_rewards.append(score)
            else:
                for child_node in node.children:
                    nodes.append(child_node)

        return all_rewards

    def update_advantages(self, node, mean, std):
        """Recursively compute normalized advantages for tree nodes."""
        if node.image:
            node.advantage = [(node.reward.to(torch.float32) - mean) / std]
            return node.advantage[0]
        else:
            advs = []
            for child in node.children:
                adv = self.update_advantages(child, mean, std)
                advs.append(adv)
            node.advantage = advs

            return (torch.cat(advs)).mean().unsqueeze(0)


    def print_tree(self, node):
        nodes = [node]
        while nodes:
            node = nodes.pop(0)
            print(node.timestep, len(node.children), node.reward, node.advantage, node.image is not None)
            for child in node.children:
                nodes.append(child)
    
    def bfs_tree(self, node):
        nodes = [(node, 0)]
        sum_depth = [0] * 10
        while nodes:
            node, depth = nodes.pop(0)
            if depth >= len(sum_depth):
                sum_depth.extend([0] * (depth + 1 - len(sum_depth)))
            if node.advantage:
                sum_depth[depth] += (torch.cat(node.advantage)).sum()
            for child_node in node.children:
                nodes.append((child_node, depth + 1))
        for i, value in enumerate(sum_depth):
            print(i, value)

    def sample(self, prompts):
        """Generate samples from the diffusion model and compute rewards/advantages."""
        self.denoiser.eval()
        expanded_prompts = []
        
        if isinstance(prompts, torch.Tensor):
            prompts = list(prompts.cpu().numpy().astype(str))
        elif isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], torch.Tensor):
            prompts = [p.item() for p in prompts]
        
        for p in prompts:
            expanded_prompts.extend([p] * self.config.sample.num_trees)
        
        global_input_latents = None
        if self.config.sample.fixed_initial_noise:
            global_input_latents = torch.randn(
                (1, 4, 64, 64),
                device=self.accelerator.device,
                dtype=self.inference_dtype,
            )
        
        pbar = tqdm(
            total=len(expanded_prompts),
            desc="Sampling",
            disable=not self.accelerator.is_local_main_process
        )


        assert self.config.sample.batch_size == 1
        root_nodes = []
        all_rewards = []
        for i in range(0, len(expanded_prompts), self.config.sample.batch_size):
            current_batch = expanded_prompts[i:i+self.config.sample.batch_size]
            with torch.no_grad():
                with self.accelerator.autocast():
                    if self.config.env.pipeline == "StableDiffusion3":
                        root_node = sd3_pipeline_with_logprob(
                            self.pipe,
                            prompt=current_batch,
                            num_inference_steps=self.config.sample.num_inference_steps,
                            guidance_scale=self.config.sample.guidance_scale,
                            height=self.config.data.height,
                            width=self.config.data.width,
                            latents=global_input_latents.repeat(len(current_batch), 1, 1, 1) if global_input_latents is not None else None,
                            noise_level=self.config.sample.noise_level,
                            tree_steps=self.tree_steps,
                            tree_k=self.config.tree.k,
                            tree_use_ode=self.config.tree.use_ode,
                        )
                        root_nodes.append(root_node)
                        rewards = self.calculate_rewards(root_node, current_batch[0])
                        all_rewards.append(torch.tensor(rewards))
                    else:
                        raise NotImplementedError(f"pipeline: {self.config.env.pipeline} not supported.")
            pbar.update(len(current_batch))

        pbar.close()
        all_rewards = torch.stack(all_rewards, dim=0).to(self.accelerator.device)
        for i in range(0, len(expanded_prompts), self.config.sample.num_trees):
            device_type = "cuda" if self.accelerator.device.type == "cuda" else "cpu"
            with torch.autocast(enabled=False, device_type=device_type):
                rewards_slice = all_rewards[i:(i + self.config.sample.num_trees)].to(torch.float32)
                mean = rewards_slice.mean()
                std = rewards_slice.std(unbiased=False) + 1e-8
                for j in range(i, i + self.config.sample.num_trees):
                    self.update_advantages(root_nodes[j], mean, std)

        all_rewards_world = self.accelerator.gather(all_rewards)

        if self.accelerator.is_main_process:
            reward_mean = all_rewards_world.mean().item()
            reward_std = all_rewards_world.std().item()
            
            self.accelerator.log(
                {
                    "reward/mean": reward_mean,
                    "reward/std": reward_std,
                },
                step=self.epoch_count
            )
            self.accelerator.print(f"[Epoch {self.epoch_count}] Reward Mean: {reward_mean:.4f}, Std: {reward_std:.4f}, Number of images: {len(all_rewards_world)}")


        all_inputs = {
            "hidden_states": [],
            "timestep": [],
            "encoder_hidden_states": [],
            "encoder_hidden_states_neg": [],
            "pooled_projections": [],
            "pooled_projections_neg": [],
            "output_latents": [],
            "log_prob": [],
            "advantages": [],
        }

        for root_node in root_nodes:
            nodes = [root_node]
            while nodes:
                node = nodes.pop(0)
                if node.output_latents:
                    all_inputs["hidden_states"].append(node.hidden_states)
                    all_inputs["timestep"].append(node.timestep)
                    all_inputs["encoder_hidden_states"].append(node.encoder_hidden_states)
                    all_inputs["encoder_hidden_states_neg"].append(node.encoder_hidden_states_neg)
                    all_inputs["pooled_projections"].append(node.pooled_projections)
                    all_inputs["pooled_projections_neg"].append(node.pooled_projections_neg)
                    all_inputs["output_latents"].append(node.output_latents)
                    all_inputs["log_prob"].append(node.log_prob)
                    all_inputs["advantages"].append(node.advantage)
                for child_node in node.children:
                    nodes.append(child_node)

        return all_inputs

    def treegrpo_update(self, log_prob, log_prob_old, advantages):
        advantages = torch.clamp(
            advantages,
            -self.config.training.adv_clip_max,
            self.config.training.adv_clip_max,
        )

        ratio = torch.exp(log_prob - log_prob_old)
        if self.global_step == 0:
            self.accelerator.print(ratio, log_prob, log_prob_old)

        # advantage > 0, ratio < 1
        #   unclipped_loss = -advantages * ratio
        #   clipped_loss = -advantages
        #   unclipped_loss
        # ad < 0, ratio > 1 + r, gradient
        # ad > 0, ratio > 1 + r, no gradient
        # ad < 0, ratio < 1 - r, no gradient
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - self.config.training.clip_range,
            1.0 + self.config.training.clip_range,
        )
        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        approx_kl = 0.5 * torch.mean((log_prob - log_prob_old) ** 2)
        clipfrac = torch.mean(
            (torch.abs(ratio - 1.0) > self.config.training.clip_range).float()
        )
        ratio_mean = torch.mean(ratio)
        ratio_min = torch.min(ratio)

        return loss, approx_kl, clipfrac, ratio_mean, ratio_min
    
    def train(self, all_inputs):
        """Run training updates on collected samples."""
        self.denoiser.train()
        total_samples = len(all_inputs["advantages"])
        self.accelerator.gradient_accumulation_steps = max(1, total_samples // 2)
        self.accelerator.print("gradient_accumulation_steps", self.accelerator.gradient_accumulation_steps)
        
        info = defaultdict(list)
        for inner_epoch in range(self.config.training.inner_epochs):
            current_all_inputs = {}

            random_indices = list(range(total_samples))
            random.shuffle(random_indices)
            for k in all_inputs:
                current_all_inputs[k] = [all_inputs[k][i] for i in random_indices]
            
            progress_bar = tqdm(
                total=total_samples,
                desc=f"Training [Epoch {self.epoch_count}, Inner {inner_epoch}]",
                disable=not self.accelerator.is_local_main_process
            )
            assert self.config.training.batch_size == 1
            for i in range(0, total_samples, self.config.training.batch_size):
                batch = {}
                batch_size = min(self.config.training.batch_size, total_samples - i)
                
                for k in current_all_inputs:
                    batch[k] = current_all_inputs[k][i:i+batch_size]
                    if isinstance(batch[k][0], torch.Tensor):
                        batch[k] = torch.cat(batch[k], dim=0)
                
                with self.accelerator.accumulate(self.denoiser):
                    with self.accelerator.autocast():
                        if self.config.env.pipeline == "StableDiffusion3":
                            if not self.config.training.enable_cfg:
                                noise_pred = self.denoiser(
                                    hidden_states=batch["hidden_states"],
                                    timestep=batch["timestep"],
                                    encoder_hidden_states=batch["encoder_hidden_states"],
                                    pooled_projections=batch["pooled_projections"],
                                    return_dict=False,
                                )[0]
                            else:
                                noise_pred_text = self.denoiser(
                                    hidden_states=batch["hidden_states"],
                                    timestep=batch["timestep"],
                                    encoder_hidden_states=batch["encoder_hidden_states"],
                                    pooled_projections=batch["pooled_projections"],
                                    return_dict=False,
                                )[0]
                                noise_pred_uncond = self.denoiser(
                                    hidden_states=batch["hidden_states"],
                                    timestep=batch["timestep"],
                                    encoder_hidden_states=batch["encoder_hidden_states_neg"],
                                    pooled_projections=batch["pooled_projections_neg"],
                                    return_dict=False,
                                )[0]
                                noise_pred = (
                                    noise_pred_uncond
                                    + self.config.sample.guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )

                            _, log_probs = flow_match_euler_discrete_step_with_logprob(
                                self.pipe.scheduler,
                                noise_pred,
                                batch["timestep"],
                                batch["hidden_states"],
                                prev_sample=batch["output_latents"][0],
                                noise_level=self.config.sample.noise_level,
                            )
                        else:
                            raise NotImplementedError(f"pipeline: {self.config.env.pipeline} not supported.")

                    if self.global_step == 0:
                        self.accelerator.print(batch["timestep"])

                    losses = []
                    approx_kls = []
                    clipfracs = []
                    ratio_means = []
                    ratio_mins = []
                    loss_sum = 0
                    assert len(log_probs) == len(batch["log_prob"][0])
                    assert len(batch["log_prob"][0]) == len(batch["advantages"][0])
                    for log_prob, old_log_prob, advantage in zip(log_probs, batch["log_prob"][0], batch["advantages"][0]):
                        loss, approx_kl, clipfrac, ratio_mean, ratio_min = self.treegrpo_update(
                            log_prob,
                            old_log_prob,
                            torch.tensor([advantage]).to(self.accelerator.device),
                        )
                        losses.append(loss)
                        approx_kls.append(approx_kl)
                        clipfracs.append(clipfrac)
                        ratio_means.append(ratio_mean)
                        ratio_mins.append(ratio_min)
                        loss_sum += loss

                    if self.config.tree.enable and self.config.tree.loss_agg == "mean":
                        loss_sum /= len(log_probs)
                    
                    info["loss"].extend(losses)
                    info["approx_kl"].extend(approx_kls)
                    info["clipfrac"].extend(clipfracs)
                    info["ratio_mean"].extend(ratio_means)
                    info["ratio_min"].extend(ratio_mins)

                    self.accelerator.backward(loss_sum)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.denoiser.parameters(),
                            max_norm=self.config.training.max_grad_norm
                        )

                        avg_info = {}
                        for k, v in info.items():
                            if k == "ratio_min":
                                avg_info[k] = torch.min(torch.stack(v))
                            else:
                                avg_info[k] = torch.mean(torch.stack(v))
                        avg_info = self.accelerator.reduce(avg_info, reduction="mean")

                        log_data = {
                            "train/loss": avg_info["loss"].item(),
                            "train/approx_kl": avg_info["approx_kl"].item(),
                            "train/clipfrac": avg_info["clipfrac"].item(),
                            "train/ratio_mean": avg_info["ratio_mean"].item(),
                            "train/global_step": self.global_step,
                            "train/epoch": self.epoch_count,
                            "train/ratio_min": avg_info["ratio_min"].item(),
                        }

                        self.accelerator.log(log_data, step=self.global_step)
                        self.global_step += 1
                        
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                progress_bar.update(batch_size)
            
            progress_bar.close()
    
    def save_model(self, epoch):
        """Save model checkpoint."""
        if self.accelerator.is_main_process:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                self.config.training.ckpt_dir,
                self.config.run_name, 
                f"checkpoint_epoch_{epoch}_{timestamp}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            unwrapped_denoiser = self.accelerator.unwrap_model(self.denoiser)
            
            save_path = os.path.join(checkpoint_dir, "unet.safetensors")
            save_file(unwrapped_denoiser.state_dict(), save_path)
            
            if self.config.lora.enable:
                lora_config_path = os.path.join(checkpoint_dir, "lora_config.json")
                unwrapped_denoiser.peft_config.save_pretrained(lora_config_path)
            
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            torch.save(self.optimizer.state_dict(), optimizer_path)
            
            self.accelerator.print(f"Saved model checkpoint at epoch {epoch} to {checkpoint_dir}")

    def run(self):
        """Full training loop"""
        self.accelerator.print(f"Starting training for {self.config.training.epochs} epochs...")
        
        for epoch, batch_prompts in enumerate(self.dataloader):
            self.epoch_count = epoch
            if self.config.tree.enable:
                if epoch % self.config.tree.tou == 0 and epoch != 0:
                    self.tree_steps = [i + self.config.tree.s for i in self.tree_steps]
                    assert self.tree_steps[-1] < self.config.sample.num_inference_steps - 1

            if epoch >= self.config.training.epochs:
                break
                
            self.accelerator.print("Sampling new images...")
            all_inputs = self.sample(batch_prompts)
            
            self.accelerator.print("Training...")
            self.train(all_inputs)

            if (epoch + 1) % self.config.training.save_ckpt_every_epoch == 0 or epoch == self.config.training.epochs - 1:
                self.save_model(epoch)
        
        self.accelerator.print("Training completed!")
        self.accelerator.end_training()

@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    if not cfg.tree.enable or not cfg.tree.use_ode:
        gradient_accumulation_steps = cfg.training.gradient_accumulation_steps * (cfg.sample.num_inference_steps - 1)
    else:
        gradient_accumulation_steps = cfg.training.gradient_accumulation_steps * (cfg.tree.w)

    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(
            project_dir=os.getcwd(),
            logging_dir=os.path.join(os.getcwd(), "tensorboard"),
        ),
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    os.makedirs(accelerator.project_configuration.logging_dir, exist_ok=True)
    trainer = RLTrainer(cfg, accelerator, log_path=cfg.run_name)
    trainer.run()

if __name__ == "__main__":
    main()
