#!/usr/bin/env bash
python data.py --save-dir data/dataset-clevr --img-dir /data/ranjaykrishna/clevr/images/val --test --image-metadata data/clevr/test_image_metadata.json --annotations data/clevr/annotations_test.json
python data.py --save-dir data/dataset-clevr --img-dir /data/ranjaykrishna/clevr/images/train --image-metadata data/clevr/train_image_metadata.json --annotations data/clevr/annotations_train.json
