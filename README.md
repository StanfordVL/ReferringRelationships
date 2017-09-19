# ReferringRelationships, User Guide

## Environment Variables

First, set environment variables

```cd ReferringRelationships```
```source set_env.sh```

## Building the dataset

Then, create the dataset for the Visual Relationship model. The script data.py will save masks for
objects and subjects in train/val/test directories that will be created in the directory --save_dir.
The script also saves numpy arrays for relationships.

```usage: data.py [-h] [--test TEST] val_split save_dir img_dir annotations image_metadata```

## Model Configuration

Change the configuration parameters in config.py

* models_dir: Directory where to save the models
* save_dir: If this is None, the script will create sub-directories into models_dir directory for each experiment, otherwise, use a path to save the model in a specific directory.
* train_data_dir: train directory that has relationships as numpy arrays and ground truth masks for objects and subjects. This directory was created with data.py above.
* val_data_dir: validation directory that has relationships as numpy arrays and ground truth masks for objects and subjects. This directory was created with data.py above.
* image_data_dir: images directory that has jpg files for images. 

## Model training

```./run.sh```

This script will train the model and save the weights in the save_dir directory. 
It will also save the configuration parameters in a params.json file, as well as the training log in a train.log file.
