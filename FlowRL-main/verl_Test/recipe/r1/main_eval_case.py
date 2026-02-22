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


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    # Set your file path here
    file_path = "/path/to/your/file.parquet"  # Replace with your actual file path
    
    local_path = copy_to_local(file_path)
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)
    
    # Save first 5 questions
    sample_data = []
    for i in range(min(5, len(dataset))):
        sample_data.append({
            "question_index": i,
            "data_source": dataset['data_source'].iloc[i],
            "question": dataset['prompt'].iloc[i][0]['content'],
            "responses": list(dataset['responses'].iloc[i])
        })
    
    output_dir = os.path.dirname(local_path)
    with open(os.path.join(output_dir, "sample_questions.json"), 'w') as f:
        json.dump(sample_data, f, indent=4)
    
    print(f"Saved first 5 questions to sample_questions.json")


if __name__ == "__main__":
    main()