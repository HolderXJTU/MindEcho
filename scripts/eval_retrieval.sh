#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python -m mindecho.eval_retrieval \
  --config configs/mindecho_nsd.yaml \
  --checkpoint checkpoints/stage1_best.pt \
  --device cuda
