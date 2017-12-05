"""Training script for referring relationships.
"""

from config import parse_args
from iterator import DiscoveryIterator, SmartIterator
from keras.optimizers import RMSprop
from models import ReferringRelationshipsModel
from utils.eval_utils import format_results_eval
from utils.visualization_utils import objdict
from utils.eval_utils import get_metrics
from utils.train_utils import get_loss_func
import json
import os


if __name__=='__main__':
    # Parse command line arguments.
    args = parse_args(evaluation=True)
    models_dir = os.path.dirname(args.model_checkpoint)
    params = objdict(json.load(open(os.path.join(models_dir, "args.json"), "r")))
    params.batch_size = args.batch_size
    params.discovery = args.discovery
    params.shuffle = False

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

    # Setup the training and validation data iterators
    if params.discovery:
        Iterator = DiscoveryIterator
    else:
        Iterator = SmartIterator
    generator = Iterator(args.data_dir, params)

    # Setup all the metrics we want to report. The names of the metrics need to
    # be set so that Keras can log them correctly.
    metrics = get_metrics(params.output_dim, args.heatmap_threshold)

    # create a new instance model
    relationships_model = ReferringRelationshipsModel(params)
    model = relationships_model.build_model()
    if params.loss_func == 'weighted':
        loss_func = get_loss_func(params.w1)
    else:
        loss_func = 'binary_crossentropy'
    model.compile(loss=[loss_func, loss_func],
                  optimizer=RMSprop(lr=0.01),
                  metrics=metrics)
    model.load_weights(args.model_checkpoint)

    # Run Evaluation.
    steps = len(generator)
    print('Total number of steps for batch size = {} : {}'.format(
        args.batch_size, steps))
    outputs = model.evaluate_generator(generator=generator,
                                       steps=steps,
                                       use_multiprocessing=args.multiprocessing,
                                       workers=args.workers)
    results = format_results_eval(model.metrics_names, outputs)
    print('Test results - ' + results)
