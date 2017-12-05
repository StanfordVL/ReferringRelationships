#!/usr/bin/env bash
CLEVR_TRAIN_IMAGES_DIR=$1
CLEVR_VAL_IMAGES_DIR=$2
python data.py --save-dir data/dataset-clevr-small --img-dir $CLEVR_VAL_IMAGES_DIR --test --image-metadata data/clevr/test_image_metadata.json --annotations data/clevr/annotations_test.json --num-images 1000 --save-images
python data.py --save-dir data/dataset-clevr-small --img-dir $CLEVR_TRAIN_IMAGES_DIR --image-metadata data/clevr/train_image_metadata.json --annotations data/clevr/annotations_train.json --num-images 10000 --save-images --val-percent 0.3
