"""Evaluation script for discovery experiment on referring relationships.

This runs 4 evaluation experiments:
    1. s = 0, o = 0
    2. s = 0, o = 1
    3. s = 1, o = 0
    4. s = 1, o = 1
Here, s and o refer to the subject-droprate and object-droprate in the
`config.py` file. So, when s = 1 and o = 1, both the subject and object
are not included in the evaluation and the model only has the predicate
to ground the subject an the object.
"""

from config import parse_args
from iterator import DiscoveryIterator
from keras.optimizers import RMSprop
from models import ReferringRelationshipsModel
from utils.eval_utils import format_results
from utils.eval_utils import get_metrics
from utils.train_utils import format_args

import logging
import os


if __name__=='__main__':
    # Parse command line arguments.
    args = parse_args(evaluation=True)

    # If the dataset does exists, alert the user.
    if not os.path.isdir(args.data_dir):
        raise ValueError('The directory %s doesn\'t exist. '
            'Exiting evaluation!' % args.data_dir)

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

    # Setup all the metrics we want to report. The names of the metrics need to
    # be set so that Keras can log them correctly.
    metrics = get_metrics(args.input_dim, args.heatmap_threshold)

    # create a new instance model
    relationships_model = ReferringRelationshipsModel(args)
    model = relationships_model.build_model()
    loss_func = 'binary_crossentropy'
    model.compile(loss=[loss_func, loss_func],
                  optimizer=RMSprop(lr=0.01),
                  metrics=metrics)
    model.load_weights(args.model_checkpoint)

    # Run Evaluation.
    for subject_droprate in [0.0, 1.0]:
        for object_droprate in [0.0, 1.0]:
            args.subject_droprate = subject_droprate
            args.object_droprate = object_droprate
            generator = DiscoveryIterator(args.data_dir, args)
            steps = len(generator)
            outputs = model.evaluate_generator(
                generator=generator, steps=steps,
                use_multiprocessing=args.multiprocessing,
                workers=args.workers)
            pre = 's_droprate: %f - o_droprate: %f - ' % (subject_droprate,
                                                          object_droprate)
            results = format_results(model.metrics_names, outputs)
            results = pre + results
            print(results)
            logging.info('='*30)
            logging.info('Test results - ' + results)
