"""This file contains all the parameters used during training and evaluating.
"""

import argparse


def parse_args():
    """Initializes a parser and reads the command line parameters.

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
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Number of dimensions in our class embeddings.')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Number of dimensions in the hidden unit.')
    parser.add_argument('--feat-map-dim', type=int, default=14,
                        help='The size of the feature map extracted from the '
                        'image.')
    parser.add_argument('--input-dim', type=int, default=224,
                        help='Size of the input image.')
    parser.add_argument('--num-subjects', type=int, default=100,
                        help='The number of subjects in the dataset.')
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

    # Parse arguments and return.
    args = parser.parse_args()
    return args


if __name__=='__main__':
    """Testing that the arguments in fact do get parsed.
    """
    args = parse_args()
    print args
