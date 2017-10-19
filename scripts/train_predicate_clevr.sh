#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py --save-dir $1 --train-data-dir data/predicate-clevr-small/train --val-data-dir data/predicate-clevr-small/val --test-data-dir data/predicate-clevr-small/test --overwrite --model baseline --log-every-batch --workers 8 --heatmap-threshold 0.5 --eval-steps 50 --reg 0.5 --lr 0.0001 --epochs 10
