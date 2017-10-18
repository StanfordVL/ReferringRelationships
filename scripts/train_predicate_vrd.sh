#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --save-dir $1 --train-data-dir data/predicate-vrd/train --val-data-dir data/predicate-vrd/val --test-data-dir data/predicate-vrd/test --overwrite --model baseline --log-every-batch --workers 8 --heatmap-threshold 0.5 --eval-steps 50 --reg 0.5
