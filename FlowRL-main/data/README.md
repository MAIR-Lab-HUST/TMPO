# Data Preparation

## Introduction

This is a collection for all data used in FlowRL paper. We detail the data source as follows:

```text
data/
├── README.md
├── math_data/
│   ├── dapo-math-17k.parquet          # Training data for math
│   ├── vaildation.parquet             # Validation data (AIME 24/25, GPQA)
│   └── test.parquet                   # Test data (AIME24/25, AMC23, MATH500, etc.)
└── code_data/
    ├── deepcoder_train-00000-of-00005.parquet    # DeepCoder training data part 1
    ├── deepcoder_train-00001-of-00005.parquet    # DeepCoder training data part 2
    ├── deepcoder_train-00002-of-00005.parquet    # DeepCoder training data part 3
    ├── deepcoder_train-00003-of-00005.parquet    # DeepCoder training data part 4
    ├── deepcoder_train-00004-of-00005.parquet    # DeepCoder training data part 5
    ├── test_codeforces.parquet                   # CodeForces test data
    ├── test_humanevalplus.parquet                # HumanEval+ test data
    ├── test_livecodebench-00000-of-00005.parquet # LiveCodeBench test data part 1
    ├── test_livecodebench-00001-of-00005.parquet # LiveCodeBench test data part 2
    ├── test_livecodebench-00002-of-00005.parquet # LiveCodeBench test data part 3
    ├── test_livecodebench-00003-of-00005.parquet # LiveCodeBench test data part 4
    ├── test_livecodebench-00004-of-00005.parquet # LiveCodeBench test data part 5
    └── test_livecodebench.json                   # LiveCodeBench test data (JSON format)
```

## Data Processing

### Option 1: Use Pre-processed Data

Download our pre-processed datasets directly from [xuekai/flowrl-data-collection](https://huggingface.co/datasets/xuekai/flowrl-data-collection).

### Option 2: Process Data from Source

**Math Data:**
- Source: [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- Processing: Use the data processing pipeline from [veRL recipe/r1](https://github.com/volcengine/verl/tree/main/recipe/r1) for math data preparation and preprocessing

**Code Data:**
- Source: DeepCoder dataset
- Processing: Use the source code from [RLLM DeepCoder](https://github.com/agentica-project/rllm/tree/deepcoder#data) for code data preparation

## Math Dataset

### Math Training Data

- **Dataset**: [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- **Description**: Training dataset for mathematical reasoning tasks

### Math Validation Data

- **AIME 24/25**: American Invitational Mathematics Examination 2024 and 2025
- **GPQA**: Graduate-level Google-Proof Q&A benchmark

### Math Test Data

- **AIME24**: American Invitational Mathematics Examination 2024
- **AIME25**: American Invitational Mathematics Examination 2025
- **AMC23**: American Mathematics Competitions 2023
- **MATH500**: MATH benchmark subset (500 problems)
- **Minerva Olympiad**: Mathematical olympiad problems from Minerva
- **Olympiad**: International Mathematical Olympiad problems

## Code Dataset

### Code Training Data

- **Dataset**: DeepCoder dataset from [RLLM](https://github.com/agentica-project/rllm/tree/deepcoder)
- **Description**: Programming problems and solutions for code generation tasks

### Code Validation Data

- **LiveCodeBench**: Subset of LiveCodeBench for validation

### Code Test Data

- **LiveCodeBench**: Live coding benchmark problems
- **CodeForces**: Competitive programming problems
- **HumanEval+**: Extended version of HumanEval benchmark

## Data Access

### Download from Hugging Face

The processed data is available at: [xuekai/flowrl-data-collection](https://huggingface.co/datasets/xuekai/flowrl-data-collection)

You can download the data using:

```bash
# Download all data
huggingface-cli download xuekai/flowrl-data-collection --repo-type dataset --local-dir ./data
```

