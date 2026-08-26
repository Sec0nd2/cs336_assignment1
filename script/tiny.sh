#!/bin/bash

MAX_LR=$1
NAME=$2

uv run python cs336_basics/llm_train.py \
  --train-data outputs/tiny_train.bin \
  --val-data outputs/tiny_valid.bin \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --batch-size 32 \
  --max-iter 5000 \
  --warm-iters 500 \
  --cosine-cycle-iters 5000 \
  --max-lr "$MAX_LR" \
  --min-lr 3e-5 \
  --log-interval 10 \
  --eval-interval 100 \
  --eval-iters 20 \
  --checkpoint-path "outputs/${NAME}.pt" \
  --device mps \
  --log-path "outputs/${NAME}.csv"