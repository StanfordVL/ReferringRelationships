# ReferringRelationships, User Guide

## Environment Variables

First, set the environment variables. You might have to change these
 depending on where you have your LD LIBRARY and PYTHON PATHS set in
 your machine.

```
cd ReferringRelationships
source set_env.sh
```

## Building the dataset

Next, create the dataset for the Visual Relationship model.
 The script `data.py` will save masks for objects and subjects
 in train/val/test directories that will be created in the directory
 `--save-dir`. The script also saves numpy arrays for relationships.

Usage:
```
python data.py [-h] [--test TEST] val_split save_dir img_dir annotations image_metadata
```

## Model Configuration

Change the configuration parameters in config.py

```
optional arguments:
  -h, --help            Show this help message and exit
  --opt                 The optimizer used during training. Currently supports
                        rms, adam, adagrad and adadelta.
  --lr                  The learning rate for training.
  --lr_decay            The learning rate decay.
  --batch-size          The batch size used in training.
  --epochs              The number of epochs to train.
  --seed                The random seed used to reproduce results.
  --overwrite           Train even if that folder already contains an existing
                        model.
  --save-dir            The location to save the model and the results.
  --models-dir          The location of the model weights
  --use-models-dir      Indicates that new models can be saved in the models
                        directory set by --models-dir.
  --save-best-only      Saves only the best model checkpoint.

  --use-subject         Boolean indicating whether to use the subjects.
  --use-predicate       Boolean indicating whether to use the predicates.
  --use-object          Boolean indicating whether to use the objects.

  --embedding-dim       Number of dimensions in our class embeddings.
  --hidden-dim          Number of dimensions in the hidden unit.
  --feat-map-dim        The size of the feature map extracted from the image.
  --input-dim           Size of the input image.
  --num-predicates      The number of predicates in the dataset.
  --num-objects         The number of objects in the dataset.
  --dropout             The dropout probability used in training.

  --train-data-dir      Location of the training data.
  --val-data-dir        Location of the validation data.
  --image-data-dir      Location of the images.
  --heatmap-threshold   The thresholds above which we consider a heatmap to
                        contain an object.
```

## Model training

```./run.sh```

This script will train the model and save the weights in the save_dir directory. 
It will also save the configuration parameters in a params.json file, as well as the training log in a train.log file.
