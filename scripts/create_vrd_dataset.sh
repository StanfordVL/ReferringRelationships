#!/usr/bin/env bash
VRD_TRAIN_IMAGES_DIR=$1
VRD_TEST_IMAGES_DIR=$2
python data.py --save-dir data/dataset-vrd --img-dir $VRD_TEST_IMAGES_DIR --test --image-metadata data/VRD/test_image_metadata.json --annotations data/VRD/annotations_test.json --save-images
python data.py --save-dir data/dataset-vrd --img-dir $VRD_TRAIN_IMAGES_DIR --image-metadata data/VRD/train_image_metadata.json --annotations data/VRD/annotations_train.json --save-images
