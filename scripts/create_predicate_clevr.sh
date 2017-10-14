#!/usr/bin/env bash
python data.py --save-dir data/predicate-clevr --img-dir /data/ranjaykrishna/images/val --test --image-metadata data/clevr/test_image_metadata.json --annotations data/clevr/annotations_test.json --dataset-type predicate --save-images
python data.py --save-dir data/predicate-clevr --img-dir /data/ranjaykrishna/images/train --image-metadata data/clevr/train_image_metadata.json --annotations data/clevr/annotations_train.json --dataset-type predicate --save-images
