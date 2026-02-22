# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import csv
import ray
import numpy as np
import hydra
import os
from tabulate import tabulate
from tqdm import tqdm

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask
import pyarrow.parquet as pq
import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    output_dir = os.path.dirname(config.data.path)
    local_path = copy_local_path_from_hdfs(config.data.path)
    ext = os.path.splitext(config.data.path)[1].lower()
    if ext == ".parquet":
        dataset = pd.read_parquet(config.data.path)
    elif ext == ".json":
        dataset = pd.read_json(config.data.path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    # Compute evaluation metrics
    # prompts = dataset[config.data.prompt_key]
    # responses = dataset['responses']  # Using the generated responses
    # data_sources = dataset[config.data.data_source_key]
    # reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    total_scores = []
    for i in tqdm(range(total), desc="Scoring responses"):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        score_lst = []
        for r in response_lst:
            try:
                score = reward_fn(r, ground_truth)
                score_lst.append(score)
            except Exception as e:
                score = reward_fn(data_source, r, ground_truth)
                score_lst.append(score)
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, f'{data_source}_pass.csv')
    
    # Prepare the row data
    # Extract the dataset name from the path
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        # 'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_samples}': pass_at_n
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

    # Convert boolean values to 0.0 or 1.0
    total_scores = [[1.0 if val else 0.0 for val in score_list] for score_list in total_scores]
    # Save the scores to results.json
    results_path = os.path.join(output_dir, f'{data_source}_results.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(total_scores, f)

# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from rllm.rewards.rl_reward import rllm_reward_fn
        return rllm_reward_fn

if __name__ == '__main__':
    main()
