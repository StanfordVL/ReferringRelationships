# Referring Relationships

![Referring Relationships model](https://cs.stanford.edu/people/ranjaykrishna/referringrelationships/model.jpg)

This repository contains code used to produce the results in the following paper:

### [Referring Relationships](https://cs.stanford.edu/people/ranjaykrishna/referringrelationships/index.html) <br/>
[Ranjay Krishna](http://ranjaykrishna.com)<sup>&dagger;</sup>, [Ines Chami](http://web.stanford.edu/~chami/)<sup>&dagger;</sup>, [Michael Bernstein](http://hci.st/msb), [Li Fei-Fei](https://twitter.com/drfeifei) <br/>
IEEE Conference on Computer Vision and Pattern Recognition ([CVPR](http://cvpr2018.thecvf.com/)), 2018 <br/>

If you are using this repository, please use the following citation:

```
@inproceedings{krishna2018referring,
  title={Referring Relationships},
  author={Krishna, Ranjay and Chami, Ines and Bernstein, Michael and Fei-Fei, Li },
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## Clone the repository and install the dependencies.

You can clone the repository and install the requirements by running the
following:

```
git clone https://github.com/stanfordvl/ReferringRelationships.git
cd ReferringRelationships
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

To download the dataset used in the project, run:

```
./scripts/download_data.sh
```

Note that we only distribute the annotations for the datasets. To
download the images for these datasets, please use the following links:

- [VRD](http://cs.stanford.edu/people/ranjaykrishna/vrd/)
- [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/)
- [Visual Genome](https://visualgenome.org)

## Model training

To train the models, you will need to create an `hdf5` dataset and then run
the following script to test and evaluate the model:

```
# For the VRD dataset.
./scripts/create_vrd_dataset.sh $LOCATION_OF_VRD_TRAIN_IMAGES $LOCATION_OF_VRD_TEST_IMAGES
./scripts/train_vrd.sh
./scripts/evaluate_vrd.sh
```

```
# For the CLEVR dataset.
./scripts/create_clevr_dataset.sh $LOCATION_OF_CLEVR_TRAIN_IMAGES $LOCATION_OF_CLEVR_VAL_IMAGES
./scripts/train_clevr.sh $LOCATION_OF_MODEL
./scripts/evaluate_clevr.sh $LOCATION_OF_MODEL
```

```
# For the Visual Genome dataset.
./scripts/create_visualgenome_dataset.sh $LOCATION_OF_VISUAL_GENOME_IMAGES
./scripts/train_visualgenome.sh $LOCATION_OF_MODEL
./scripts/evaluate_visualgenome.sh $LOCATION_OF_MODEL
```

This script will train the model and save the weights in the `--save-dir`
directory.  It will also save the configuration parameters in a 
`params.json` file and log events in `train.log`.

However, if you decide that you want more control over the training or
evaluation scripts, check out the instructions below.

## Customized dataset creation

The script `data.py` will save masks for objects and subjects
in train/val/test directories that will be created in the directory
`--save-dir`. The script also saves numpy arrays for relationships.

The script has the following command line arguments to modify the dataset
pre-processing:

```
  -h, --help            show this help message and exit
  --test                When true, the data is not split into training and
                        validation sets
  --val-percent         Fraction of images in validation split.
  --save-dir            where to save the ground truth masks, this Location
                        where dataset should be saved.
  --img-dir             Location where images are stored.
  --annotations         Json with relationships for each image.
  --image-metadata      Image metadata json file.
  --image-dim           The size the images should be saved as.
  --output-dim          The size the predictions should be saved as.
  --seed                The random seed used to reproduce results.
  --num-images          The random seed used to reproduce results.
  --save-images         Use this flag to specify that the images should also
                        be saved.
  --max-rels-per-image  Maximum number of relationships per image.
```

## Customized Training.

The model can be trained by calling `python train.py` with the following command
line arguments to modify your training:

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

## Customized evaluation

The evaluations can be run using `python evaluate.py` with the following options:

```
  -h, --help            show this help message and exit
  --batch-size          The batch size used in training.
  --seed                The random seed used to reproduce results.
  --workers             Number workers used to load the data.
  --heatmap-threshold   The thresholds above which we consider a heatmap to
                        contain an object.
  --model-checkpoint    The model to evaluate.
  --data-dir            Location of the data to evluate with.
```

## Customized discovery evaluation.

The discovery based experiments can be run by setting the following flags 
during training and using `python evaluate_discovery.py` when evaluating.

```
  --discovery           Used when we run the discovery experinent where
                        objects are dropped during training.
  --always-drop-file    Location of list of objects that should always be
                        dropped.
  --subject-droprate    Rate at which subjects are dropped.
  --object-droprate     Rate at which objects are dropped.
  --model-checkpoint    The model to evaluate.
  --data-dir            Location of the data to evluate with.

```

## Contributing.

We welcome everyone to contribute to this reporsitory. Send us a pull request.

## License:

The code is under the MIT license. Check `LICENSE` for details.
