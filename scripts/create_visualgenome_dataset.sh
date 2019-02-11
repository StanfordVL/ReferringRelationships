#!/usr/bin/env bash
VISUAL_GENOME_IMAGES_DIR=$1
python data.py --save-dir data/dataset-visualgenome --img-dir $VISUAL_GENOME_IMAGES_DIR --test --image-metadata data/VisualGenome/test_image_metadata.json --annotations data/VisualGenome/annotations_test.json --save-images
python data.py --save-dir data/dataset-visualgenome --img-dir $VISUAL_GENOME_IMAGES_DIR --image-metadata data/VisualGenome/train_image_metadata.json --annotations data/VisualGenome/annotations_train.json --save-images
