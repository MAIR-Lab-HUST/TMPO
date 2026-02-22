from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import hydra
import ray

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
import os
import json
from datetime import datetime


@ray.remote
def evaluate_single_item(config, data_source, response_lst, reward_data):
    """Evaluate responses for a single question and return individual scores"""
    reward_fn = get_custom_reward_fn(config)
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return score_lst, np.mean(score_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    # Hardcoded paths - modify as needed
    flowrl_path = "/fs-computility/plm/shared/zhuxuekai/reasoning_flow/outputs/qwen_32b/gfn_is_qwen_32B_0629_global_step_200/test-output-16.parquet"
    grpo_path = "/fs-computility/plm/shared/zhuxuekai/reasoning_flow/outputs/qwen_32b/GRPO_global_step_200/test-output-16.parquet"
    
    output_dir = "plot/diversity/comparison_flowrl_vs_grpo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    flowrl_local = copy_to_local(flowrl_path)
    grpo_local = copy_to_local(grpo_path)
    
    flowrl_dataset = pd.read_parquet(flowrl_local)
    grpo_dataset = pd.read_parquet(grpo_local)
    
    print(f"FlowRL dataset size: {len(flowrl_dataset)}")
    print(f"GRPO dataset size: {len(grpo_dataset)}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)
    
    # Evaluate both datasets
    total = min(len(flowrl_dataset), len(grpo_dataset))
    
    # Create remote tasks for FlowRL
    flowrl_tasks = [evaluate_single_item.remote(
        config, 
        flowrl_dataset[config.data.data_source_key].iloc[i], 
        flowrl_dataset[config.data.response_key].iloc[i], 
        flowrl_dataset[config.data.reward_model_key].iloc[i]
    ) for i in range(total)]
    
    # Create remote tasks for GRPO
    grpo_tasks = [evaluate_single_item.remote(
        config, 
        grpo_dataset[config.data.data_source_key].iloc[i], 
        grpo_dataset[config.data.response_key].iloc[i], 
        grpo_dataset[config.data.reward_model_key].iloc[i]
    ) for i in range(total)]
    
    # Process FlowRL results
    flowrl_results = []
    with tqdm(total=total, desc="Evaluating FlowRL") as pbar:
        while len(flowrl_tasks) > 0:
            done_ids, flowrl_tasks = ray.wait(flowrl_tasks)
            for result_id in done_ids:
                scores, avg_score = ray.get(result_id)
                flowrl_results.append((scores, avg_score))
                pbar.update(1)
    
    # Process GRPO results
    grpo_results = []
    with tqdm(total=total, desc="Evaluating GRPO") as pbar:
        while len(grpo_tasks) > 0:
            done_ids, grpo_tasks = ray.wait(grpo_tasks)
            for result_id in done_ids:
                scores, avg_score = ray.get(result_id)
                grpo_results.append((scores, avg_score))
                pbar.update(1)
    
    # Find samples where FlowRL succeeds and GRPO fails
    comparison_samples = []
    num_samples_to_extract = 10  # Change this to 5 or 10 as needed
    
    for i in range(total):
        flowrl_scores, flowrl_avg = flowrl_results[i]
        grpo_scores, grpo_avg = grpo_results[i]
        
        # Define success as average score > 0
        flowrl_success = flowrl_avg > 0
        grpo_success = grpo_avg > 0
        
        # Check if FlowRL succeeds and GRPO fails
        if flowrl_success and not grpo_success:
            sample = {
                "question_index": i,
                "data_source": flowrl_dataset[config.data.data_source_key].iloc[i],
                "question": flowrl_dataset[config.data.prompt_key].iloc[i][0]['content'],
                "ground_truth": flowrl_dataset[config.data.reward_model_key].iloc[i]['ground_truth'],
                "flowrl_responses": list(flowrl_dataset[config.data.response_key].iloc[i]),
                "grpo_responses": list(grpo_dataset[config.data.response_key].iloc[i]),
                "flowrl_scores": flowrl_scores,
                "grpo_scores": grpo_scores,
                "flowrl_avg_score": flowrl_avg,
                "grpo_avg_score": grpo_avg,
                "timestamp": datetime.now().isoformat()
            }
            comparison_samples.append(sample)
            print(f"Found sample {len(comparison_samples)}: Question {i} - FlowRL: {flowrl_avg:.3f}, GRPO: {grpo_avg:.3f}")
            
            if len(comparison_samples) >= num_samples_to_extract:
                break
    
    # Save results
    if comparison_samples:
        output_file = os.path.join(output_dir, f"flowrl_correct_grpo_wrong_{len(comparison_samples)}_samples.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_samples, f, indent=4, ensure_ascii=False)
        
        print(f"\n=== Results ===")
        print(f"Found {len(comparison_samples)} samples where FlowRL succeeds and GRPO fails")
        print(f"Saved to: {output_file}")
        
        # Print summary statistics
        flowrl_avg_scores = [s['flowrl_avg_score'] for s in comparison_samples]
        grpo_avg_scores = [s['grpo_avg_score'] for s in comparison_samples]
        
        print(f"FlowRL average scores: {np.mean(flowrl_avg_scores):.3f} ± {np.std(flowrl_avg_scores):.3f}")
        print(f"GRPO average scores: {np.mean(grpo_avg_scores):.3f} ± {np.std(grpo_avg_scores):.3f}")
        
        # Data source distribution
        data_source_counts = {}
        for sample in comparison_samples:
            ds = sample['data_source']
            data_source_counts[ds] = data_source_counts.get(ds, 0) + 1
        print(f"Data source distribution: {data_source_counts}")
    else:
        print("No samples found where FlowRL succeeds and GRPO fails")


if __name__ == "__main__":
    main()