from datasets import load_dataset, Dataset
import json
import os

data_path = "data/math_data/validation.parquet"

def robust_load_dataset(data_path):
    ext = os.path.splitext(data_path)[-1]
    try:
        if ext == ".parquet":
            return load_dataset("parquet", data_files=data_path)["train"]
        elif ext == ".json":
            return load_dataset("json", data_files=data_path)["train"]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        print(f"⚠️ Standard loading failed: {e}")
        # fallback to from_list if it's json
        if ext == ".json":
            try:
                with open(data_path, "r") as f:
                    data = json.load(f)
                print("✅ Loaded JSON with Dataset.from_list fallback")
                return Dataset.from_list(data)
            except Exception as e2:
                raise RuntimeError(f"Fallback JSON load failed: {e2}")
        else:
            raise RuntimeError(f"Parquet load failed and no fallback available: {e}")

# 加载数据（支持 parquet/json fallback）
train_data = robust_load_dataset(data_path)

# 打印 Dataset 信息
print("Train Dataset:")
print(train_data)

# 打印总样本数和首个样本结构
print(f"\nTotal examples: {len(train_data)}\n")
print("First sample:")
print(train_data[0])

# 遍历前5条样本
for i in range(min(5, len(train_data))):
    print(f"\n--- Sample {i} ---")
    sample = train_data[i]

    print("data_source:", sample.get('data_source', 'N/A'))
    print("ability:", sample.get('ability', 'N/A'))

    reward_model = sample.get("reward_model", {})
    print("ground_truth:", reward_model.get("ground_truth", "N/A"))

    prompt = sample.get("prompt", [])
    if isinstance(prompt, list) and len(prompt) > 0:
        print("prompt:\n", prompt[0].get("content", "N/A"))
    else:
        print("prompt: N/A")

    extra_info = sample.get("extra_info", {})
    print("index:", extra_info.get("index", "N/A"))
    print("raw_problem:", extra_info.get("raw_problem", "N/A"))