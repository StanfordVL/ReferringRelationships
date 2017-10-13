#!/usr/bin/env bash
python data.py --save-dir data/dataset-vrd --img-dir /data/chami/VRD/sg_dataset/sg_test_images --test --image-metadata data/VRD/test_image_metadata.json --annotations data/VRD/annotations_test.json
python data.py --save-dir data/dataset-vrd --img-dir /data/chami/VRD/sg_dataset/sg_train_images --image-metadata data/VRD/train_image_metadata.json --annotations data/VRD/annotations_train.json
