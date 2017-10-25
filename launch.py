import argparse
import os
import random
import subprocess


parser = argparse.ArgumentParser(description='Run the retrieval model with varying parameters.')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--nruns', type=int, default=50)
parser.add_argument('--workers', type=str, default='8')
parser.add_argument('--epochs', type=str, default='100')
parser.add_argument('--models-dir', type=str,
                    default='/data/chami/ReferringRelationships/models/VRD/10_24_2017/baseline')
parser.add_argument('--train-data-dir', type=str,
                    default='/data/ranjaykrishna/ReferringRelationships/data/dataset-vrd/train')
parser.add_argument('--val-data-dir', type=str,
                    default='/data/ranjaykrishna/ReferringRelationships/data/dataset-vrd/val')
parser.add_argument('--test-data-dir', type=str,
                    default='/data/ranjaykrishna/ReferringRelationships/data/dataset-vrd/test')
parser.add_argument('--model', type=str, default='baseline')
parser.add_argument('--categorical-predicate', type=bool, default=True)
parser.add_argument('--num-predicates', type=str, default='70')
parser.add_argument('--num-objects', type=str, default='100')
args = parser.parse_args()

for _ in range(args.nruns):
    params = {
        'lr': '%.4f' % random.uniform(0.1, 10.0),
        'dropout': '%.1f' % random.uniform(0.0, 0.5),
        'weight-decay': '%.1f' % random.uniform(0.0001, 0.001),
        'opt': 'rms',
        'hidden-dim': 'rms',
        'feat-map-layer activation': 'activation_40',
        'feat-map-dim': 14,
        'heatmap-threshold': 0.5,
        'nb-conv-im-map': 5,
        'conv-im-kernel': 1,
        'nb-conv-att-map': 3,
        'conv-predicate-kernel': 3,
        'lr-reduce-rate': 0.8,
        'batch-size': 128,
        'use-predicate': 1,
        'w1': 1.,
        'iterator-type': 'smart',
        'patience': 4,
        'embedding-dim': 128
    }
    arguments = ' '.join(['--' + k + ' ' + str(params[k]) for k in params])
    train = 'CUDA_VISIBLE_DEVICES=' + args.gpu + ' python train.py --use-models-dir --model ' + args.model + ' --epochs ' + args.epochs + ' --workers ' + args.workers
    train += ' --models-dir ' + args.models_dir + ' --train-data-dir ' + args.train_data_dir + ' --val-data-dir ' + args.val_data_dir + ' --test-data-dir ' + args.test_data_dir
    if args.categorical_predicate:
        train += ' --categorical-predicate'
    train += ' --num-predicates ' + args.num_predicates + ' --num-objects ' + args.num_objects
    train += ' ' + arguments
    print('\n' +'*'*89 + '\n')
    print(train)
    # subprocess.call(train, shell=True)
