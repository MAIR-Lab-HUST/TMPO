

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
import os
import json
from openai import OpenAI
from datetime import datetime

def evaluate_gpt_diversity(output_dir, dataset, api_key, use_sampling=False, sample_size=10):
    """GPT diversity evaluation with 1-5 scale for all 16 responses"""
    client = OpenAI(
                    api_key=api_key,
                    base_url="https://lonlie.plus7.plus/v1"
                )
    
    import random
    import json
    import os

    
    # Get AIME problems
    aime_indices = [i for i, ds in enumerate(dataset['data_source']) 
                   if ds in ['aime2024', 'aime2025'] and len(dataset['responses'].iloc[i]) >= 10]
    
    if not aime_indices:
        return {}
    
    # Use all problems or sample
    if use_sampling:
        eval_indices = random.sample(aime_indices, min(sample_size, len(aime_indices)))
        print(f"Using sampling: evaluating {len(eval_indices)} out of {len(aime_indices)} AIME problems...")
    else:
        eval_indices = aime_indices
        print(f"Evaluating ALL {len(eval_indices)} AIME problems...")
    
    gpt_scores = []
    gpt_logs = []
    
    for count, idx in enumerate(eval_indices, 1):
        try:
            problem = dataset['prompt'].iloc[idx][0]['content']
            responses = list(dataset['responses'].iloc[idx])  # All 16 responses
            data_source = dataset['data_source'].iloc[idx]
            
            # Format all responses
            formatted_responses = []
            for i, response in enumerate(responses):
                formatted_responses.append(f"Response {i+1}: {response[:300]}{'...' if len(response) > 300 else ''}")
            
            prompt = f"""You are evaluating the DIVERSITY of solution approaches for a mathematics competition problem. Focus on detecting even SUBTLE differences in methodology that indicate different problem-solving strategies.

PROBLEM:
{problem}

16 SOLUTION ATTEMPTS:
{chr(10).join(formatted_responses)}

EVALUATION CRITERIA - Rate diversity from 1 to 5:

**Score 1 - Minimal Diversity:**
- 14+ responses use essentially identical approaches
- Same mathematical setup, same variable choices, same solution path
- Only trivial differences (arithmetic, notation, wording)
- Indicates very low exploration/diversity in the generation process

**Score 2 - Low Diversity:** 
- 11-13 responses use the same main approach  
- 1-2 alternative approaches appear but are rare
- Minor variations within the dominant method (different substitutions, orderings)
- Some exploration but heavily biased toward one strategy

**Score 3 - Moderate Diversity:**
- 7-10 responses use the most common approach
- 2-3 distinct alternative approaches present
- Noticeable variation in problem setup or mathematical techniques
- Balanced mix showing reasonable exploration

**Score 4 - High Diversity:**
- 4-6 responses use the most common approach  
- 3-4 distinct solution strategies well-represented
- Multiple mathematical techniques and problem framings
- Strong evidence of diverse exploration strategies

**Score 5 - Maximum Diversity:**
- No single approach dominates (≤3 responses use same method)
- 4+ distinctly different solution strategies
- Wide variety of mathematical techniques and creative approaches
- Excellent exploration and generation diversity

**KEY INDICATORS TO LOOK FOR:**
- Different variable definitions/setups
- Alternative mathematical techniques (algebraic vs geometric vs analytical)
- Different problem decomposition strategies  
- Varied approaches to constraints/equations
- Different computational pathways even with same high-level method

**IMPORTANT:** Even if many responses are incomplete or incorrect, focus on the DIVERSITY of the attempted approaches, not solution quality.

Return ONLY a number from 1 to 5:"""
            
            # Call GPT
            gpt_response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0, 
                max_tokens=5
            )
            
            raw_response = gpt_response.choices[0].message.content.strip()
            score = int(raw_response)
            score = max(1, min(5, score))  # Ensure score is 1-5
            
            gpt_scores.append(score)
            log_entry = {
                "problem_idx": idx,
                "data_source": data_source,
                "gpt_score": score,
                "openai_response": gpt_response.model_dump(),
                "timestamp": datetime.now().isoformat()
            }
            gpt_logs.append(log_entry)
            
            print(f"[{count}/{len(eval_indices)}] Problem {idx} ({data_source}): GPT score = {score}")
            
        except Exception as e:
            print(f"[{count}/{len(eval_indices)}] GPT eval failed for problem {idx}: {e}")
            score = 3  # Default to moderate diversity
            gpt_scores.append(score)
            
            # Log the error
            error_log = {
                "problem_idx": idx,
                "data_source": dataset['data_source'].iloc[idx],
                "error": str(e),
                "fallback_score": score,
                "timestamp": datetime.now().isoformat()
            }
            gpt_logs.append(error_log)
    
    # Save detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "sampled" if use_sampling else "all"
    log_filename = os.path.join(output_dir, f"gpt_diversity_logs_{mode}_{timestamp}.jsonl")
    
    with open(log_filename, 'w', encoding='utf-8') as f:
        for log_entry in gpt_logs:
            try:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed to serialize log entry: {e}")
                # Fallback: save minimal info
                minimal_log = {
                    "problem_idx": log_entry.get("problem_idx", "unknown"),
                    "error": "serialization_failed",
                    "details": str(e)
                }
                f.write(json.dumps(minimal_log) + "\n")
    
    avg_score = sum(gpt_scores) / len(gpt_scores) if gpt_scores else 0
    
    print(f"\n=== GPT Diversity Evaluation Results ===")
    print(f"Mode: {'Sampled' if use_sampling else 'All AIME problems'}")
    print(f"Problems evaluated: {len(gpt_scores)}")
    print(f"GPT Diversity (1-5 scale): {avg_score:.2f}")
    print(f"Score distribution: {dict((i, gpt_scores.count(i)) for i in range(1, 6))}")
    print(f"Detailed logs saved to: {log_filename}")
    
    return {
        "gpt_diversity_aime_1to5": avg_score,
        "gpt_diversity_scores": gpt_scores,
        "gpt_problems_evaluated": len(gpt_scores),
        "gpt_total_aime_problems": len(aime_indices),
        "gpt_evaluation_mode": mode,
        "gpt_log_file": log_filename,
        "gpt_score_distribution": {str(i): gpt_scores.count(i) for i in range(1, 6)}
    }


@ray.remote
def process_item(config, data_source, response_lst, reward_data):
    reward_fn = get_custom_reward_fn(config)
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)

    # Create remote tasks
    remote_tasks = [process_item.remote(config, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)

    metric_dict = {}

    # Calculate AIME diversity metric (Equation 14)
    # aime_scores = data_source_reward.get('aime2024', []) + data_source_reward.get('aime2025', [])
    # aime_success_counts = [score * 16 for score in aime_scores]  # S_i = score * 16 responses
    # aime_solved_problems = [s for s in aime_success_counts if s >= 1]  # Apply indicator I(S_i >= 1)
    # aime_diversity_score = sum(aime_solved_problems) / len(aime_solved_problems) if aime_solved_problems else 0
    
    # print(f"AIME Diversity (Equation 14): {aime_diversity_score:.3f}")
    # print(f"AIME Total questions: {len(aime_scores)}")
    # print(f"AIME Questions with ≥1 success: {len(aime_solved_problems)}")
    # print(f"AIME Total successful trajectories: {sum(aime_solved_problems):.0f}")
    
    # for data_source, rewards in data_source_reward.items():
    #     metric_dict[f"test_score/{data_source}"] = np.mean(rewards)
    
    # # Add AIME diversity to metrics
    # metric_dict["diversity_equation14_aime"] = aime_diversity_score


    output_dir = os.path.dirname(local_path)
    base_output_dir = os.path.dirname(local_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_output_dir, "diveristy_evaluation_results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    # GPT Diversity Evaluation
    gpt_api_key = "sk-cySsnCdGiI8jysaHtQjbuOkoCEch2JDxYbrLLbyJX7sJm8Mu"  # Replace with your key
    gpt_results = evaluate_gpt_diversity(results_dir, dataset, gpt_api_key)
    metric_dict.update(gpt_results)
    
    json_output_path = os.path.join(results_dir, "diversity_evaluation_results.json")
    with open(json_output_path, "w") as f:
        json.dump(metric_dict, f, indent=4)

    print(metric_dict)

    


if __name__ == "__main__":
    main()