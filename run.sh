#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --use-models-dir --lr 0.001 --epochs 200 --models-dir /data/chami/ReferringRelationships/models/10_06_2017 --heatmap-threshold 0.5 --hidden-dim 128 --feat-map-dim 14 --model ssn --train-data-dir data/VRD/overfit/test/ --val-data-dir data/VRD/overfit/test/ --batch-size 1
