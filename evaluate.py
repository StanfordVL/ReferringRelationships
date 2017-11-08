"""Training script for referring relationships.
"""

from keras.models import load_model

from config import parse_args
from iterator import DiscoveryIterator, SmartIterator
from keras.optimizers import RMSprop
from models import ReferringRelationshipsModel
from utils.eval_utils import format_results
from utils.eval_utils import get_metrics
from utils.train_utils import format_args

import json
import logging
import os


if __name__=='__main__':
    # Parse command line arguments.
    args = parse_args(evaluation=True)

    # If the dataset does exists, alert the user.
    if not os.path.isdir(args.data_dir):
        raise ValueError('The directory %s doesn\'t exist. '
            'Exiting evaluation!' % args.save_dir)

    # Make sure the dataset and images exist.
    for hdf5_file in [os.path.join(args.data_dir, 'images.hdf5'),
                      os.path.join(args.data_dir, 'dataset.hdf5')]:
        if not os.path.exists(hdf5_file):
            raise ValueError('The dataset %s doesn\'t exist. '
                'Exiting evaluation!' % hdf5_file)

    # Setup logging.
    logfile = os.path.join(args.model_dir, 'evaluation.log')
    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=logfile)

    # Store the arguments used in this training process.
    logging.info(format_args(args))

    # Setup the training and validation data iterators
    if args.discovery:
        Iterator = DiscoveryIterator
    else:
        Iterator = SmartIterator
    args.droprate = 0.0
    generator = Iterator(args.data_dir, args)
    logging.info('Evaluating on {} samples'.format(generator.samples))

    # Setup all the metrics we want to report. The names of the metrics need to
    # be set so that Keras can log them correctly.
    metrics = get_metrics(args.input_dim, args.heatmap_threshold)

    # create a new instance model
    relationships_model = ReferringRelationshipsModel(args)
    model = relationships_model.build_model()
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                  optimizer=RMSprop(lr=0.01),
                  metrics=metrics)
    model.load_weights(args.model_checkpoint)

    # Run Evaluation.
    steps = len(generator)
    outputs = model.evaluate_generator(generator=generator,
                                       steps=steps,
                                       use_multiprocessing=args.multiprocessing,
                                       workers=args.workers)
    results = format_results(model.metrics_names, outputs)
    print('Test results - ' + results)
    logging.info('='*30)
    logging.info('Test results - ' + results)
