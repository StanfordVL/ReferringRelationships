#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-models-dir --lr 0.0001 --epochs 200 --models-dir /data/chami/ReferringRelationships/models/10_09_2017 --heatmap-threshold 0.5 --hidden-dim 256 --feat-map-dim 14 --model ssn --train-data-dir /data/chami/VRD/10_09_2017/train/ --val-data-dir /data/chami/VRD/10_09_2017/val --batch-size 128 --dropout 0.2 --model-checkpoint /data/chami/ReferringRelationships/models/10_09_2017/8/model00-1.02.h5

