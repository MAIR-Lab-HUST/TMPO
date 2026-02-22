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
Preprocess the dataset to parquet format
"""

import os
from datasets import load_dataset, concatenate_datasets
from functools import partial

from verl.utils.hdfs_io import copy, makedirs
import argparse

dapo_prompt_prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
dapo_prompt_suffix = "\n\nRemember to put your answer on its own line after \"Answer:\"."

def example_map_fn(example, idx, process_fn, data_source, ability, split):
    question, solution = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question
        }],
        "ability": ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx
        }
    }
    return data

def dapo_example_map_fn(example, idx, process_fn, data_source, ability, split):
    question, raw_problem, solution = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question
        }],
        "ability": ability,
        "reward_model": {
            "style": "rule-lighteval/MATH_v2",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            "raw_problem": raw_problem,
            'index': idx
        }
    }
    return data


def build_aime2024_dataset():

    def process_aime2024(example):
        processed_prompt = dapo_prompt_prefix + example["Problem"] + dapo_prompt_suffix
        return processed_prompt, example["Problem"], str(example["Answer"])

    data_source = 'Maxwell-Jia/AIME_2024'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(dapo_example_map_fn,
                     process_fn=process_aime2024,
                     data_source="aime2024",
                     ability="MATH",
                     split=None)
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset

def build_aime2025_dataset():

    def process_aime2025(example):
        processed_prompt = dapo_prompt_prefix + example["problem"] + dapo_prompt_suffix
        return processed_prompt, example["problem"], str(example["answer"])

    data_source = 'yentinglin/aime_2025'# hf 
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(dapo_example_map_fn,
                     process_fn=process_aime2025,
                     data_source="aime2025",
                     ability="MATH",
                     split=None)
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset

def build_amc23_dataset():

    def process_amc23(example):
        processed_prompt = dapo_prompt_prefix + example["question"] + dapo_prompt_suffix
        return processed_prompt, example["question"], str(example["answer"])

    data_source = 'zwhe99/amc23'# local path
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    map_fn = partial(dapo_example_map_fn,
                     process_fn=process_amc23,
                     data_source="amc23",
                     ability="MATH",
                     split=None)
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset

def build_MATH500_dataset():

    def process_math500(example):
        processed_prompt = dapo_prompt_prefix + example["problem"] + dapo_prompt_suffix
        return processed_prompt, example["problem"], str(example["answer"])

    data_source = 'HuggingFaceH4/MATH-500'# hf 
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    map_fn = partial(dapo_example_map_fn,
                     process_fn=process_math500,
                     data_source="math500",
                     ability="MATH",
                     split=None)
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset

def build_minerva_dataset():

    def process_minerva(example):
        processed_prompt = dapo_prompt_prefix + example["question"] + dapo_prompt_suffix
        return processed_prompt, example["question"], str(example["answer"])

    data_source = 'math-ai/minervamath'# hf 
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    map_fn = partial(dapo_example_map_fn,
                     process_fn=process_minerva,
                     data_source="minerva",
                     ability="MATH",
                     split=None)
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset

def build_olympiadbench_dataset():

    def process_olympiadbench(example):

        processed_prompt = dapo_prompt_prefix + example["question"] + dapo_prompt_suffix
        final_answer = example["final_answer"]
        if isinstance(final_answer, list) and len(final_answer) > 0:
            answer_str = str(final_answer[0])
        else:
            answer_str = str(final_answer)
        answer_str = answer_str.strip()

        return processed_prompt, example["question"], answer_str

    data_source = 'math-ai/olympiadbench'# hf 
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    map_fn = partial(dapo_example_map_fn,
                     process_fn=process_olympiadbench,
                     data_source="olympiadbench",
                     ability="MATH",
                     split=None)
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset



def build_gpqa_dimond_dataset():
    import random
    GPQA_QUERY_TEMPLATE = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"

    def process_gpqa_diamond(example):
        choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(A=choices[0],
                                                  B=choices[1],
                                                  C=choices[2],
                                                  D=choices[3],
                                                  Question=example["Question"])
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    data_source = 'Idavidrein/gpqa'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "gpqa_diamond", split="train")
    map_fn = partial(example_map_fn,
                     process_fn=process_gpqa_diamond,
                     data_source=data_source,
                     ability="Math",
                     split="test")
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_cnmo2024_dataset():

    def process_cnmo2024(example):
        return example["question"], example["answer"]

    data_source = 'opencompass/LiveMathBench'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset_en = load_dataset(data_source, "v202412_CNMO_en", split="test")
    map_fn_en = partial(example_map_fn,
                        process_fn=process_cnmo2024,
                        data_source='opencompass/cnmo2024_en',
                        ability="Math",
                        split="test")
    dataset_en = dataset_en.map(map_fn_en, with_indices=True, remove_columns=dataset_en.column_names)

    dataset_zh = load_dataset(data_source, "v202412_CNMO_cn", split="test")
    map_fn_zh = partial(example_map_fn,
                        process_fn=process_cnmo2024,
                        data_source='opencompass/cnmo2024_zh',
                        ability="Math",
                        split="test")
    dataset_zh = dataset_zh.map(map_fn_zh, with_indices=True, remove_columns=dataset_zh.column_names)

    dataset = concatenate_datasets([dataset_en, dataset_zh])
    return dataset


def build_livecodebench_dataset():
    import json, pickle, zlib, base64

    def process_livecodebench(example):
        # Construct Query Prompt
        # From https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/prompts/code_generation.py#L140
        query_prompt = (
            "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
            f"Question: {example['question_content']}\n\n")
        if example["starter_code"]:
            query_prompt += (
                "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
                f"```python\n{example['starter_code']}\n```")
        else:
            query_prompt += (
                "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
                f"```python\n# YOUR CODE HERE\n```")

        # Construct test cases
        public_test_cases = json.loads(example["public_test_cases"])
        try:
            private_test_cases = json.loads(example["private_test_cases"])
        except:
            private_test_cases = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(example["private_test_cases"].encode("utf-8")))))
        full_test_cases = public_test_cases + private_test_cases

        metadata = json.loads(example["metadata"])
        test_cases = {
            "inputs": [t["input"] for t in full_test_cases],
            "outputs": [t["output"] for t in full_test_cases],
            "fn_name": metadata.get("func_name", None),
        }
        text_cases_compressed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")
        return query_prompt, text_cases_compressed

    data_source = 'livecodebench/code_generation_lite'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    
    dataset = load_dataset('json', data_dir="/root/workspace/reasoning_flow/data/livecodebench/code_generation_lite", split="test")
    # R1 Evaluation use LiveCodeBench 24.08-25.01

    # dataset = dataset.filter(lambda line: "2024-08-00T00:00:00" <= line["contest_date"] < "2025-01-00T00:00:00")

    from datetime import datetime
    start_date = datetime(2024, 8, 1)
    end_date = datetime(2025, 1, 1)
    dataset = dataset.filter(lambda line: start_date <= line["contest_date"] < end_date)

    map_fn = partial(example_map_fn,
                     process_fn=process_livecodebench,
                     data_source=data_source,
                     ability="Code",
                     split="test")

    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names, num_proc=8)
    return dataset


TASK2DATA = {
    "aime2024": build_aime2024_dataset, 
    "aime2025": build_aime2025_dataset, 
    "amc23": build_amc23_dataset, 
    "math500": build_MATH500_dataset, 
    "minerva": build_minerva_dataset, 
    "olympiadbench": build_olympiadbench_dataset,
    # "gpqa_diamond": build_gpqa_dimond_dataset, # *2 -> prompt
    # "cnmo2024": build_cnmo2024_dataset,
    # "livecodebench": build_livecodebench_dataset, # *1
}
SUPPORTED_TASKS = TASK2DATA.keys()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/r1')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--tasks', default="all")

    args = parser.parse_args()

    if args.tasks.lower() == "all":
        args.tasks = SUPPORTED_TASKS
    else:
        args.tasks = [task.strip() for task in args.tasks.split(',') if task.strip()]
        for task in args.tasks:
            if task not in SUPPORTED_TASKS:
                raise NotImplementedError(f"{task} has not been supported.")

    # datasets = []
    # for task in args.tasks:
    #     for _ in range(16):
    #         datasets.append(TASK2DATA[task]())
    # for task in args.tasks:
    #     if task in ["aime2024", "aime2025"]:
    #         for i in range(8):
    #             datasets.append(TASK2DATA[task]())
    #     elif task == "gpqa_diamond":
    #         for i in range(2):
    #             datasets.append(TASK2DATA[task]())
    #     else:
    #         datasets.append(TASK2DATA[task]())
    datasets = []
    for task in args.tasks:
        datasets.append(TASK2DATA[task]())

    test_dataset = concatenate_datasets(datasets)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
