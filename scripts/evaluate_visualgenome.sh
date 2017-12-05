#!/usr/bin/env bash
SAVE_DIR=$1
python evaluate.py --model-checkpoint $SAVE_DIR --data-dir data/dataset-visualgenome/test --heatmap-threshold 0.5 --batch-size 16
