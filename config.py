"""This file contains all the parameters used during training and evaluating.
"""

import argparse
import numpy as np


def parse_training_args(parser):
    """Add args used for training only.

    Args:
        parse: An argparse object.
    """

    # Session parameters.
    parser.add_argument('--opt', type=str, default='rms',
                        help='The optimizer used during training. Currently'
                        ' supports rms, adam, adagrad and adadelta.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='The number of epochs to train.')
    parser.add_argument('--train-steps-per-epoch', type=int, default=-1,
                        help='The total number of steps (batches of samples) '
                        'to yield from generator before declaring one epoch '
                        'finished and starting the next epoch.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Train even if that folder already contains '
                        'an existing model.')
    parser.add_argument('--log-every-batch', action='store_true',
                        help='Logs every batch when used. Otherwise it '
                        'logs every epoch.')
    parser.add_argument('--val-steps-per-epoch', type=int, default=-1,
                        help='Number of steps to yield from validation '
                        'generator at the end of every epoch.')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle the dataset.')
    parser.add_argument('--categorical-predicate', action='store_true',
                        default=False,
                        help='wheteher to return indexes or masks for the '
                        'iterator, should only be used ssas models.')
    parser.add_argument('--use-internal-loss', action='store_true', default=False,
                        help='Whether to add intermediate losses in the ssas model')
    parser.add_argument('--internal-loss-weight', type=float, default=1.,
                        help='The co-officient of the internal loss of the '
                        'subject and object. Must use --model ssas, '
                        '--use-internal-loss.')
    parser.add_argument('--loss-func', type=str, default='basic',
                         help='[basic, weighted] cross entropy loss. '
                         'Must set --w1 when using --loss-func weighted.')
    parser.add_argument('--w1', type=float, default=2.,
                        help='The coefficient to use on the positive '
                        'examples in the CE loss')

    # Learning rate parameters.
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The learning rate for training.')
    parser.add_argument('--patience', type=int, default=0,
                        help='The number of epochs to wait if val loss is '
                        'increasing and decrease the learning rate.')
    parser.add_argument('--lr-reduce-rate', type=float, default=0.1,
                        help='Multiple to reduce the learning rate by.')

    # Model parameters.
    parser.add_argument('--model', type=str, default='ssas',
                        help='Indicates which model to use: '
                        '[co-occurrence, vrd, ssas] Check paper for details.')
    parser.add_argument('--use-subject', type=int, default=1,
                        help='1/0 indicating whether to use the subjects.')
    parser.add_argument('--use-predicate', type=int, default=1,
                        help='1/0 indicating whether to use the predicates.')
    parser.add_argument('--use-object', type=int, default=1,
                        help='1/0 indicating whether to use the objects.')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                        help='Number of dimensions in the hidden unit.')
    parser.add_argument('--embedding-dim', type=int, default=512,
                         help='Number of dimensions in the embedding unit.')
    parser.add_argument('--cnn', type=str, default='resnet',
                        help='The pretrained cnn architecture [resnet, vgg] '
                        'to use to extract image features')
    parser.add_argument('--feat-map-dim', type=int, default=14,
                        help='The size of the feature map extracted from the '
                        'image.')
    parser.add_argument('--feat-map-layer', type=str, default='activation_40',
                        help='The feature map to use '
                        'for resnet: (activation_40 for 14x14 and '
                        'activation_22 for 28x28) for vgg: '
                        '(block3_conv4 for 56x56, block4_conv4 for 28x28 and '
                        'block5_conv4 for 14x14)')
    parser.add_argument('--input-dim', type=int, default=224,
                        help='Size of the input image.')
    parser.add_argument('--output-dim', type=int, default=14,
                        help='Size of the output predictions.')
    parser.add_argument('--num-predicates', type=int, default=70,
                        help='The number of predicates in the dataset.')
    parser.add_argument('--num-objects', type=int, default=100,
                        help='The number of objects in the dataset.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='The dropout probability used in training.')
    parser.add_argument('--nb-conv-im-map', type=int, default=0,
                        help='Number of convolution layers to use '
                        'to learn image feature maps')
    parser.add_argument('--nb-conv-att-map', type=int, default=4,
                        help='Number of convolution layers to use to shift '
                        'attention in ssas model')
    parser.add_argument('--conv-im-kernel', type=int, default=1,
                        help='The kernel size when using convolutions in '
                        'the ssas model to compute image feature maps')
    parser.add_argument('--conv-predicate-kernel', type=int, default=5,
                        help='The kernel size when using convolutions in '
                        'the ssas model to move heatmaps')
    parser.add_argument('--conv-predicate-channels', default=10, type=int,
                        help='Number of channels to use in convolution filters that shift attention')
    parser.add_argument('--iterations', default=2, type=int,
                        help='The number of iterations to finetune the heatmaps.')
    parser.add_argument('--attention-conv-kernel', default=3, type=int,
                        help='kernel size for the attention module.')
    parser.add_argument('--refinement-conv-kernel', default=3, type=int,
                        help='kernel size for the attention module.')
    parser.add_argument('--finetune-cnn', action='store_true',
                        help='Finetune the pretrained cnn networks.')

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
    parser.add_argument('--model-checkpoint', type=str, default=None,
                        help='The location of the last checkpoint to reload')

    # Data parameters.
    parser.add_argument('--train-data-dir', type=str,
                        default='/data/chami/VRD/09_20_2017/train/',
                        help='Location of the training data.')
    parser.add_argument('--val-data-dir', type=str,
                        default='/data/chami/VRD/09_20_2017/val/',
                        help='Location of the validation data.')
    parser.add_argument('--test-data-dir', type=str,
                        default='/data/chami/VRD/09_20_2017/test/',
                        help='Location of the validation data.')


def parse_evaluation_args(parser):
    """Add args used for evaulating a model only.

    Args:
        parse: An argparse object.
    """
    parser.add_argument('--model-checkpoint', type=str,
                        help='The model to evaluate.')
    parser.add_argument('--data-dir', type=str,
                        default='data/pred-vrd/test/',
                        help='Location of the data to evluate with.')

def parse_args(evaluation=False):
    """Initializes a parser and reads the command line parameters.

    Args:
        evaluation: Boolean set to true if we are evaluating instead of
            training.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """
    parser = argparse.ArgumentParser(description='Referring Relationships.')

    # Session parameters.
    parser.add_argument('--batch-size', type=int, default=128,
                        help='The batch size used in training.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='The random seed used to reproduce results.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number workers used to load the data.')

    # Eval parameters.
    parser.add_argument('--heatmap-threshold', type=float, nargs='+',
                        default=[0.3, 0.5, 0.6],
                        help='The thresholds above which we consider '
                        'a heatmap to contain an object.')

    # Discovery Experiment.
    parser.add_argument('--discovery', action='store_true', default=False,
                        help='Used when we run the discovery experinent '
                        'where objects are dropped during training.')
    parser.add_argument('--always-drop-file', type=str, default=None,
                        help='Location of list of objects that should always '
                        'be dropped.')
    parser.add_argument('--subject-droprate', type=float, default=0.3,
                        help='Rate at which subjects are dropped.')
    parser.add_argument('--object-droprate', type=float, default=0.3,
                        help='Rate at which objects are dropped.')


    # Grab the other parameters.
    if evaluation:
        parse_evaluation_args(parser)
    else:
        parse_training_args(parser)

    args = parser.parse_args()

    if not evaluation:
        # Verify that we have at least one of the following flags set:
        args.use_subject = args.use_subject > 0
        args.use_predicate = args.use_predicate > 0
        args.use_object = args.use_object > 0
        if not (args.use_subject or args.use_predicate or args.use_object):
            raise ValueError('At least one of the 3 components of the '
                'relationship should be included in training.')

    # set the random seed.
    np.random.seed(args.seed)

    # Set flags for multiprocessing.
    args.multiprocessing = args.workers > 1

    return args


if __name__=='__main__':
    """Testing that the arguments in fact do get parsed.
    """
    args = parse_args()
    args = args.__dict__
    print("Arguments:")
    for k in args:
        print('\t%15s:\t%s' % (k, args[k]))
