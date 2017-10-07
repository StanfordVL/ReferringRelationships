#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --use-models-dir --lr 0.0001 --epochs 50 --models-dir /data/chami/ReferringRelationships/models/10_06_2017 --embedding-dim 256 --hidden-dim 512 --heatmap-threshold 0.5
