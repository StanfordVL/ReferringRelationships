#!/usr/bin/env bash
python data.py --save-dir data/dataset-vrd --img-dir /data/ranjaykrishna/vrd/images/val --test --image-metadata data/vrd/test_image_metadata.json --annotations data/vrd/annotations_test.json
python data.py --save-dir data/dataset-vrd --img-dir /data/ranjaykrishna/vrd/images/train --image-metadata data/vrd/train_image_metadata.json --annotations data/vrd/annotations_train.json
