"""Find the model that performs the best on validation set.
"""

import argparse
import json
import numpy as np
import os


def print_scores(model, imax, s_score, o_score, thresh):
    print('\nModel: {}'.format(model))
    print('\nEpoch: {}'.format(imax))
    print("\n{} {}".format(args.s_metric.format(thresh), s_score))
    print("\n{} {}".format(args.o_metric.format(thresh), o_score))
    print('\n'+'*'*50)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Reads log file to find the best model')
    parser.add_argument('--models-dir', type=str,
                        help='Directory where all the models are saved.')
    parser.add_argument('--s-metric', type=str,
                        default='val_subject_precision_{}:',
                        help='Name of metric to use when comparing models.')
    parser.add_argument('--o-metric', type=str,
                        default='val_object_precision_{}:',
                        help='Name of metric to use when comparing models.')
    args = parser.parse_args()

    directory = args.models_dir
    max_s, max_o = 0, 0
    max_model = None
    max_sum = 0
    max_epoch = 0
    best_thresh = 0
    for model_idx in next(os.walk(directory))[1]:
        try:
            data = open(os.path.join(
                directory, model_idx, 'train.log')).readlines()
            params = json.load(
                open(os.path.join(directory, model_idx,"args.json")))
        except IOError:
            print(model_idx)
            continue
        data = [x.split() for x in data if len(x.split())>1]
        data = [x for x in data if x[0]=="lr:"]

        if len(data)>0:
            i = data[0].index(args.s_metric.format(params["heatmap_threshold"][0])) + 1
            j = data[0].index(args.o_metric.format(params["heatmap_threshold"][0])) + 1
            x = np.array([[x[i][:-1], x[j][:-1]] for x in data]).astype(np.float)
            imax = np.argmax(x.sum(axis=1))
            current_sum = x.sum(axis=1)[imax]
            print_scores(model_idx, imax, x[imax][0], x[imax][1], params["heatmap_threshold"][0])
            if current_sum > max_sum:
                max_sum = current_sum
                max_s, max_o = x[imax][0], x[imax][1]
                max_model = model_idx
                max_epoch = imax
                best_thresh = params["heatmap_threshold"][0]
    print('\n'+'*'*50 + 'BEST MODEL' +'*'*50)
    print_scores(max_model, max_epoch, max_s, max_o, best_thresh)
