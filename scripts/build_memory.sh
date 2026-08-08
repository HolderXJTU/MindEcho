#!/usr/bin/env bash

python tools/build_external_memory.py \
  --image_features memory/image_features.pt \
  --text_features memory/text_features.pt \
  --output memory/memory_clip.pt
