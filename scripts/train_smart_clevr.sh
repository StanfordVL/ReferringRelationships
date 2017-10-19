#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py --save-dir $1 --train-data-dir data/dataset-clevr-small/train --val-data-dir data/dataset-clevr-small/val --test-data-dir data/dataset-clevr-small/test --overwrite --model baseline --log-every-batch --workers 8 --heatmap-threshold 0.5 --eval-steps 50 --reg 0.5 --iterator-type smart --epochs 10
