#!/usr/bin/env bash
VISUAL_GENOME_IMAGES_DIR=$1
python data.py --save-dir data/dataset-vrd --img-dir $VISUAL_GENOME_IMAGES_DIR --test --image-metadata data/VRD/test_image_metadata.json --annotations data/VRD/annotations_test.json --save-images
python data.py --save-dir data/dataset-vrd --img-dir $VISUAL_GENOME_IMAGES_DIR --image-metadata data/VRD/train_image_metadata.json --annotations data/VRD/annotations_train.json --save-images
