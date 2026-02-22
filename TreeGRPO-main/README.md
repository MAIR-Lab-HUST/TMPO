# TreeGRPO (ICLR 2026)

[![Paper](https://img.shields.io/badge/arXiv-2512.08153-b31b1b.svg)](https://www.arxiv.org/abs/2512.08153)
[![Project Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://treegrpo.github.io)

TreeGRPO introduces tree-advantage estimation for GRPO in online RL post-training of diffusion models. This repo provides the core codebase for the SD3.5-medium training loop using HPSv2.

## Setup

```sh
conda create -n treegrpo python=3.11
conda activate treegrpo
pip install -r requirements.txt
```

## Training

```sh
accelerate launch --num_processes=8 train.py \
  --config-name base run_name=sd3-5m_treegrpo \
  sample.num_prompts=2 sample.num_trees=1 \
  tree.w=4 tree.k=2
```

## Key parameters

- `tree.w`: tree window size.
- `tree.k`: number of branches per split.
- `tree.s`: step increment when shifting the tree window.
- `tree.tou`: frequency (in epochs) to shift the tree window.
- `sample.num_prompts`: number of prompts per gpu/process per epoch
- `sample.num_trees`: number of trees per prompt

* Total sampled images and training samples per epoch per gpu/process in TreeGRPO:
  - **Images per epoch** = $(\text{tree.k})^{\text{tree.w}} \times$ sample.num_trees $\times$ sample.num_prompts
  - **Training samples per epoch** = $\frac{(\text{tree.k})^{\text{tree.w}} - 1}{\text{tree.k} - 1} \times$ sample.num_trees $\times$ sample.num_prompts

## HPSv2 checkpoint

Download `HPS_v2.1_compressed.pt` from the [HPSv2](https://huggingface.co/xswu/HPSv2) project and place it at `hps_ckpt/HPS_v2.1_compressed.pt`.

## Data

We use the prompts from [HPDv2](https://huggingface.co/datasets/ymhao/HPDv2/tree/main) dataset for training which can be found in `prompts.txt`.

## Acknowledgements

We thank the authors of [DDPO](https://github.com/kvablack/ddpo-pytorch), [DanceGRPO](https://github.com/XueZeyue/DanceGRPO) and [FlowGRPO](https://github.com/yifan123/flow_grpo) for open-sourcing their code and resources.

## Citation

If you use this repository, please consider citing the TreeGRPO paper:

```bibtex
@inproceedings{
    ding2026treegrpo,
    title={TreeGRPO: Tree-Advantage GRPO for Online RL Post-Training of Diffusion Models},
    author={Ding, Zheng and Ye, Weirui},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026}
}
```
