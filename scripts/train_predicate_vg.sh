#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --save-dir $1 --train-data-dir data/predicate-visualgenome/train --val-data-dir data/predicate-visualgenome/val --test-data-dir data/predicate-visualgenome/test --overwrite --model baseline --log-every-batch --workers 8 --heatmap-threshold 0.5 --eval-steps 50 --reg 0.5 --epochs 10
