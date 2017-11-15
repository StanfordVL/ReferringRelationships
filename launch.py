import argparse
import os
import random
import subprocess
import numpy as np

parser = argparse.ArgumentParser(description='Run the ReferringRelationships model with varying parameters.')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--nruns', type=int, default=50)
parser.add_argument('--workers', type=str, default='8')
parser.add_argument('--epochs', type=str, default='50')
parser.add_argument('--models-dir', type=str,
                    default='/data/chami/ReferringRelationships/models/VRD/10_24_2017/baseline')
parser.add_argument('--train-data-dir', type=str,
                    default='/data/ranjaykrishna/ReferringRelationships/data/dataset-vrd/train')
parser.add_argument('--val-data-dir', type=str,
                    default='/data/ranjaykrishna/ReferringRelationships/data/dataset-vrd/val')
parser.add_argument('--test-data-dir', type=str,
                    default='/data/ranjaykrishna/ReferringRelationships/data/dataset-vrd/test')
parser.add_argument('--model', type=str, default='baseline')
parser.add_argument('--categorical-predicate', action='store_true')
parser.add_argument('--use-internal-loss', action='store_true')
parser.add_argument('--num-predicates', type=str, default='70')
parser.add_argument('--num-objects', type=str, default='100')
parser.add_argument('--use-predicate', type=str, default='1', help='1/0 indicating whether to use the predicates.')
args = parser.parse_args()


for _ in range(args.nruns):
    params = {
        'lr': 0.0001, 
        'patience': 3,
        'lr-reduce-rate': 0.7,
        'dropout': 0, 
        'opt': "rms", 
        'batch-size': 64, 
        'hidden-dim': 1024, 
        'embedding-dim': np.random.choice([256, 512, 1024]),
        'input-dim': 224, 
        'output-dim': 14, 
        'cnn': 'resnet', 
        'feat-map-layer': 'activation_40',
        'feat-map-dim': 14,
        'nb-conv-im-map': 0,
        'conv-im-kernel': 1,
        'nb-conv-att-map': np.random.choice([4, 5, 6]),
        'conv-predicate-kernel': np.random.choice([3, 4, 5, 7]),
        'heatmap-threshold': 0.5, 
        'conv-predicate-channels': np.random.choice([10, 20, 50]),
        'w1': 7.5, 
        'loss-func': 'weighted',
        'internal-loss-weight': np.random.choice([1., 2., 5.]),
        'iterations': np.random.choice([2, 3])
    }
    arguments = ' '.join(['--' + k + ' ' + str(params[k]) for k in params])
    train = 'CUDA_VISIBLE_DEVICES=' + args.gpu + ' python train.py --use-models-dir --model ' + args.model + ' --epochs ' + args.epochs + ' --workers ' + args.workers
    train += ' --models-dir ' + args.models_dir + ' --train-data-dir ' + args.train_data_dir + ' --val-data-dir ' + args.val_data_dir + ' --test-data-dir ' + args.test_data_dir
    if args.categorical_predicate:
        train += ' --categorical-predicate'
    if args.use_internal_loss:
        train += ' --use-internal-loss'
    train += ' --num-predicates ' + args.num_predicates + ' --num-objects ' + args.num_objects + ' --use-predicate ' + args.use_predicate
    train += ' ' + arguments
    print('\n' +'*'*89 + '\n')
    print(train)
    subprocess.call(train, shell=True)
