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
parser.add_argument('--num-predicates', type=str, default='70')
parser.add_argument('--num-objects', type=str, default='100')
#parser.add_argument('--use-subject', type=int, default=1, help='1/0 indicating whether to use the subjects.')
parser.add_argument('--use-predicate', type=str, default='1', help='1/0 indicating whether to use the predicates.')
#parser.add_argument('--use-object', type=int, default=1, help='1/0 indicating whether to use the objects.')
args = parser.parse_args()


for _ in range(args.nruns):
    params = {
        'lr': '%.4f' % random.uniform(0.0001, 0.01),
        'patience': np.random.choice([3, 5]),
        'lr-reduce-rate': '%.1f' % random.uniform(0.5, 0.9),
        'dropout': '%.1f' % random.uniform(0.0, 0.5),
        'opt': np.random.choice(['rms','adam']), 
        'batch-size': np.random.choice([64, 128, 256]),
        'hidden-dim': np.random.choice([128, 256, 512]), 
        'input-dim': 224, 
        'feat-map-layer': 'activation_40',
        'feat-map-dim': 14,
        'nb-conv-im-map': np.random.choice([1, 2]),
        'conv-im-kernel': 1,
        'nb-conv-att-map': np.random.choice([1, 2, 3, 4]),
        'conv-predicate-kernel': np.random.choice([3, 5, 7]),
        'heatmap-threshold': 0.5,
        'val-steps-per-epoch': 50,
        #'w1': 1.,
    }
    arguments = ' '.join(['--' + k + ' ' + str(params[k]) for k in params])
    train = 'CUDA_VISIBLE_DEVICES=' + args.gpu + ' python train.py --use-models-dir --model ' + args.model + ' --epochs ' + args.epochs + ' --workers ' + args.workers
    train += ' --models-dir ' + args.models_dir + ' --train-data-dir ' + args.train_data_dir + ' --val-data-dir ' + args.val_data_dir + ' --test-data-dir ' + args.test_data_dir
    if args.categorical_predicate:
        train += ' --categorical-predicate'
    train += ' --num-predicates ' + args.num_predicates + ' --num-objects ' + args.num_objects + ' --use-predicate ' + args.use_predicate
    train += ' ' + arguments
    print('\n' +'*'*89 + '\n')
    print(train)
    subprocess.call(train, shell=True)
