"""This file contains all the parameters used during training and evaluating.
"""

import argparse
import numpy as np

def parse_args():
    """Initializes a parser and reads the command line parameters.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """
    parser = argparse.ArgumentParser(description='Referring Relationships.')

    # Session parameters.
    parser.add_argument('--opt', type=str, default='rms',
                        help='The optimizer used during training. Currently'
                        ' supports rms, adam, adagrad and adadelta.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='The learning rate for training.')
    parser.add_argument('--lr_decay', type=float, default=0,
                        help='The learning rate decay.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='The batch size used in training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The number of epochs to train.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='The random seed used to reproduce results.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Train even if that folder already contains '
                        'an existing model.')

    # Locations read and written to in the filesystem.
    parser.add_argument('--save-dir', type=str, default=None,
                        help='The location to save the model and the results.')
    parser.add_argument('--models-dir', type=str,
                        default='/data/chami/ReferringRelationships/09_20_2017',
                        help='The location of the model weights')
    parser.add_argument('--use-models-dir', action='store_true',
                        help='Indicates that new models can be saved in the'
                        ' models directory set by --models-dir.')
    parser.add_argument('--save-best-only', action='store_true',
                        help='Saves only the best model checkpoint.')

    # Model parameters.
    parser.add_argument('--use-subject', type=int, default=1,
                        help='1/0 indicating whether to use the subjects.')
    parser.add_argument('--use-predicate', type=int, default=1,
                        help='1/0 indicating whether to use the predicates.')
    parser.add_argument('--use-object', type=int, default=1,
                        help='1/0 indicating whether to use the objects.')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Number of dimensions in our class embeddings.')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Number of dimensions in the hidden unit.')
    parser.add_argument('--feat-map-dim', type=int, default=14,
                        help='The size of the feature map extracted from the '
                        'image.')
    parser.add_argument('--input-dim', type=int, default=224,
                        help='Size of the input image.')
    parser.add_argument('--num-predicates', type=int, default=70,
                        help='The number of predicates in the dataset.')
    parser.add_argument('--num-objects', type=int, default=100,
                        help='The number of objects in the dataset.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='The dropout probability used in training.')

    # Data parameters.
    parser.add_argument('--train-data-dir', type=str,
                        default='/data/chami/VRD/09_20_2017/train/',
                        help='Location of the training data.')
    parser.add_argument('--val-data-dir', type=str,
                        default='/data/chami/VRD/09_20_2017/val/',
                        help='Location of the validation data.')
    parser.add_argument('--image-data-dir', type=str,
                        default='/data/chami/VRD/sg_dataset/sg_train_images/',
                        help='Location of the images.')

    # Eval parameters.
    parser.add_argument('--heatmap-threshold', type=float, nargs='+',
                        default=[0.3, 0.5, 0.6],
                        help='The thresholds above which we consider '
                        'a heatmap to contain an object.')

    # Parse arguments.
    args = parser.parse_args()

    # set the random seed.
    np.random.seed(args.seed)

    # Verify that we have at least one of the following flags set:
    args.use_subject = args.use_subject > 0
    args.use_predicate = args.use_subject > 0
    args.use_object = args.use_subject > 0
    if not (args.use_subject or args.use_predicate or args.use_object):
        raise ValueError('At least one of the 3 components of the '
            'relationship should be included in training.')

    return args


if __name__=='__main__':
    """Testing that the arguments in fact do get parsed.
    """
    args = parse_args()
    args = args.__dict__
    print "Arguments:"
    for k in args:
        print '\t%15s:\t%s' % (k, args[k])
