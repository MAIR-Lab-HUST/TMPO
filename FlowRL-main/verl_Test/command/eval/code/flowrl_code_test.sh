#!/bin/bash
# Default values

BASE_DIR="<YOUR_BASE_PATH>"
MODEL_PATH="<YOUR_MODEL_PATH>"
OUTPUT_DIR="$<YOUR_OUTPUT_PATH>"
TP_SIZE=2
N_SAMPLES=16
MAX_RESPONSE_LEN=8192
PROMPT_LEN=2048
N_GPUS_PER_NODE=8
N_NODES=1

# Fixed datasets and their corresponding paths
declare -A DATASETS
DATASETS["codeforces"]="${BASE_DIR}/data/code_data/test_codeforces.parquet"
DATASETS["humanevalplus"]="${BASE_DIR}/data/code_data/test_humanevalplus.parquet"
DATASETS["livecodebench"]="${BASE_DIR}/data/code_data/test_livecodebench.json"

# Loop through all predefined datasets
for DATASET_NAME in "${!DATASETS[@]}"; do
    DATA_PATH="${DATASETS[$DATASET_NAME]}"
    OUTPUT_PATH="${OUTPUT_DIR}/${DATASET_NAME}-output-${N_SAMPLES}.parquet"

    echo "Running generation for ${DATASET_NAME}"
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=$N_NODES \
        trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
        data.path="$DATA_PATH" \
        data.prompt_key=prompt \
        data.batch_size=1024 \
        data.n_samples=$N_SAMPLES \
        data.output_path="$OUTPUT_PATH" \
        model.path="$MODEL_PATH" \
        rollout.temperature=0.6 \
        rollout.top_p=0.95 \
        rollout.prompt_length=$PROMPT_LEN \
        rollout.response_length=$MAX_RESPONSE_LEN \
        rollout.tensor_model_parallel_size=$TP_SIZE \
        rollout.gpu_memory_utilization=0.7 \
        rollout.max_num_batched_tokens=65536
done

FILES=(
    "humanevalplus-output-16.parquet"
    "codeforces-output-16.parquet"
    "livecodebench-output-16.json"
)

PROMPT_KEY=prompt
RESPONSE_KEY=responses
REWARD_FUNC_PATH=recipe/r1/reward_score.py
REWARD_FUNC_NAME=reward_func

# Evaluation loop
for FILE in "${FILES[@]}"; do
    echo "Evaluating $FILE..."
    python3 -m recipe.r1.main_eval_for_code \
        data.path="${OUTPUT_DIR}/${FILE}" \
        data.prompt_key=$PROMPT_KEY \
        data.response_key=$RESPONSE_KEY \
        custom_reward_function.path=$REWARD_FUNC_PATH \
        custom_reward_function.name=$REWARD_FUNC_NAME
done

RESULTS_JSON_PATH=$OUTPUT_DIR/codeforces_results.json
python benchmark/cf_elo_calc.py --results_path "$RESULTS_JSON_PATH" --pass_n 16