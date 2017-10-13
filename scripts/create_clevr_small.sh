#!/usr/bin/env bash
python data.py --save-dir data/dataset-clevr-small --img-dir /data/ranjaykrishna/clevr/images/val --test --image-metadata data/clevr/test_image_metadata.json --annotations data/clevr/annotations_test.json --num-images 1000 --save-images
python data.py --save-dir data/dataset-clevr-small --img-dir /data/ranjaykrishna/clevr/images/train --image-metadata data/clevr/train_image_metadata.json --annotations data/clevr/annotations_train.json --num-images 10000 --save-images --val-percent 0.3
