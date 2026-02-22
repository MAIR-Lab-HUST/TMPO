#!/bin/bash

DATASET_NAME=xuekai/flowrl-data-collection

huggingface-cli download $DATASET_NAME \
  --repo-type dataset \
  --resume-download \
  --local-dir data/$DATASET_NAME \
  --local-dir-use-symlinks False