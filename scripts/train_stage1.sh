#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python -m mindecho.train_stage1 \
  --config configs/mindecho_nsd.yaml \
  --device cuda
