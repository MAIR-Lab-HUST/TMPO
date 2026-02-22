#!/bin/bash

MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

huggingface-cli download $MODEL_NAME \
  --repo-type model \
  --resume-download \
  --local-dir downloads/$MODEL_NAME \
  --local-dir-use-symlinks False \
  --exclude *.pth