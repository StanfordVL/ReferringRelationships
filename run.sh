#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-models-dir --lr 0.0001 --epochs 100 --models-dir /data/chami/ReferringRelationships/models/10_09_2017 --heatmap-threshold 0.3 --hidden-dim 512 --feat-map-dim 14 --model ssn --train-data-dir /data/chami/VRD/10_09_2017/train/ --val-data-dir /data/chami/VRD/10_09_2017/val --batch-size 128

