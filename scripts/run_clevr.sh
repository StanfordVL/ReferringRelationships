#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py --save-dir $1 --train-data-dir data/dataset-clevr/train --val-data-dir data/dataset-clevr/val --test-data-dir data/dataset-clevr/test --overwrite --model baseline --log-every-batch --workers 8 --heatmap-threshold 0.5
