import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from diffusers import FluxPipeline
from diffusers.utils import is_torch_xla_available
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from safetensors.torch import load_file
import argparse
from diffusers import FluxTransformer2DModel, AutoencoderKL
import json
import random
import numpy as np
from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
from fastvideo.utils.sampling_utils import flow_grpo_step, dance_grpo_step, run_sample_step, sd3_time_shift, dpm_step
from tqdm.auto import tqdm
from diffusers.image_processor import VaeImageProcessor
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
from fastvideo.utils.logging_ import main_print


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

def sanitize_filename(text, max_length=200):
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', text)
    return sanitized[:max_length].rstrip() or "untitled"

def distributed_setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if XLA_AVAILABLE:
        xm.set_rng_state(seed)

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def sample_reference_model(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    caption,
    timesteps_train, # index
    global_step,
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1).to(device)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []  
    all_multi_rewards = {}
    all_image_ids = []

    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
                generator=torch.Generator(device=device).manual_seed(args.seed)   
            )
    
    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (len(batch_idx), IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
        input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)
        image_ids = prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device, torch.bfloat16)
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress", disable=not dist.is_initialized() or dist.get_rank() != 0)

        determistic = [True] * sample_steps
        for i in timesteps_train:
            determistic[i] = False

        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents_new,
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                batch_text_ids,
                image_ids,
                grpo_sample,
                determistic=determistic,
            )
        
        all_image_ids.append(image_ids)
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        vae.enable_tiling()
        
        image_processor = VaeImageProcessor(16)
        rank = int(os.environ["RANK"])

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, h, w, 8)
                latents = (latents / 0.3611) + 0.1159
                image = vae.decode(latents, return_dict=False)[0]
                decoded_image = image_processor.postprocess(image)

        image_dir = f"{args.output_dir}/images_{args.SDE_sampling_start_step}_{args.SDE_sampling_end_step}_{args.sampling_steps}"
        os.makedirs(image_dir, exist_ok=True)
        decoded_image[0].save(f"{image_dir}/flux_{global_step}_{rank}.png")

        latent_dir = f"{args.output_dir}/latents_{args.SDE_sampling_start_step}_{args.SDE_sampling_end_step}_{args.sampling_steps}"
        os.makedirs(latent_dir, exist_ok=True)
        latent = latents[0].cpu().numpy()
        np.save(f"{latent_dir}/flux_{global_step}_{rank}.npy", latent)

        print(f"Generated image and latent for step {global_step}, rank {rank}")
        

def main(args):
    # init
    rank, local_rank, world_size = distributed_setup()
    set_seed(args.seed + rank)
    if rank == 0:
        for key, value in vars(args).items():
            print(f"{key}: {value}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=1,
        num_workers=4,
        drop_last=True,
    )

    loader = sp_parallel_dataloader_wrapper(
        dataloader,
        "cuda",
        1,
        1,
        1,
    )

    # load the model
    transformer = FluxTransformer2DModel.from_pretrained(
        "data/flux",
        subfolder="transformer",
        torch_dtype = torch.float32
    )

    transformer = transformer.to("cuda")

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to("cuda")

    if args.model_path:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    # infer
    for step in range(int(args.total_num / world_size)):
        (
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            caption,
        ) = next(loader)

        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds)
        text_ids = repeat_tensor(text_ids)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")
        
        timesteps_train = [ti for ti in range(args.sampling_steps)]
        timesteps_train = timesteps_train[args.SDE_sampling_start_step:args.SDE_sampling_end_step]

        sample_reference_model(
            args,
            "cuda", 
            transformer,
            vae,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            caption,
            timesteps_train,
            step,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux Inference")
    ### File parameters
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="data/flux",
                        help="Path to the Flux model directory")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the Flux model checkpoint")
    parser.add_argument("--data_json_path", type=str, default="data/rl_embeddings/videos2caption.json",
                        help="Path to the JSON file containing dataset annotations")
    parser.add_argument("--output_dir", type=str, default="./output_flux",
                        help="Directory to save generated images embeddings")


    ### Inference parameters
    parser.add_argument("--baseline", action='store_true', default=False,
                        help="Use baseline model settings")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random number generation")
    parser.add_argument("--SDE_sampling_start_step", type=int, default=0)
    parser.add_argument("--SDE_sampling_end_step", type=int, default=13)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--num_latent_t", type=int, default=1,
                        help="Number of latent time steps")
    parser.add_argument("--cfg", type=float, default=0.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_generations", type=int, default=1,
                        help="Number of generations per prompt")
    parser.add_argument("--init_same_noise", action='store_true', default=False,
                        help="Initialize with the same noise for all samples")
    parser.add_argument("--total_num", type=int, default=100,
                        help="Total number of samples to generate")
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--drop_last_sample",
        action="store_true",
        default=False,
        help="whether to drop the last sample in the batch if it is not complete",
    )
    #################### Progressive ####################
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="progressive",
        choices=["progressive", "random", "decay", "exp_decay"],
        help="sample timesteps strategy for grpo",
    )
    #################### Sampling ####################
    parser.add_argument(
        "--dpm_algorithm_type",
        type=str,
        default="null",
        choices=["null", "dpmsolver", "dpmsolver++"],
    )
    parser.add_argument(
        "--dpm_apply_strategy",
        type=str,
        default="post",
        choices=["post", "all"],
    )
    parser.add_argument(
        "--dpm_post_compress_ratio",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--dpm_solver_order",
        type=int,
        default=2,
    )
    parser.add_argument(
       "--dpm_solver_type",
        type=str,
        default="heun",
        choices=["heun", "midpoint"]
    )
    parser.add_argument(
        "--flow_grpo_sampling",
        action="store_true",
        default=False,
        help="whether to use flow grpo sampling",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )

    args = parser.parse_args()

    main(args)
