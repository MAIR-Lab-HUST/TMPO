#!/bin/bash
set -x 

BACKEND="fsdp"  
LOCAL_DIR="<YOUR_WEIGHT_PATH>"
TARGET_DIR="<YOUR_TARGET_PATH>"

PYTHONPATH=. python scripts/model_merger.py merge \
  --backend $BACKEND \
  --local_dir $LOCAL_DIR \
  --target_dir $TARGET_DIR

