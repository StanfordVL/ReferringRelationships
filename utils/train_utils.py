"""Utility functions for training.
"""

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.optimizers import RMSprop, Adam, Adagrad, Adadelta

import cv2
import numpy as np
import os


def save_weights(img_path, model_path, save_path, target_size=(224, 224)):
    model = load_model(model_path)
    img = load_img(os.path.join(img_path, target_size=target_size))
    img = img.reshape((1,) + img.shape)
    # pred = model.predict


def visualize_weights(orig_image, pred, input_dim, epoch, name, res_dir):
    # saving attention heatmaps for one example
    pred = pred.reshape(input_dim, input_dim, 1)
    pred = 1. * (pred > 0.9)
    image_pred = np.zeros((input_dim, input_dim, 3))
    image_pred += pred * 255
    attention_viz = cv2.addWeighted(orig_image, 0.6, image_pred, 0.4, 0)
    cv2.imwrite(os.path.join(res_dir, 'attention-' + name + '-' + str(epoch) + '.png'), attention_viz)


def get_opt(opt, lr, lr_decay):
    """Initializes the opt that we want to train with.

    Args:
        opt: A string representation of the opt chosen by the user.
        lr: The learning rate used to initialize the optimizer.
        lr_decay: The learning rate decay used to initialize the optimizer.

    Returns:
        The keras optimizer we are going to use.
    """
    if opt=="rms":
        return RMSprop(lr=lr,decay=lr_decay)
    elif opt=="adam":
        return Adam(lr=lr,decay=lr_decay)
    elif opt=="adagrad":
        return Adagrad(lr=lr,decay=lr_decay)
    elif opt=="adadelta":
        return Adadelta(lr=lr,decay=lr_decay)
    else:
        raise ValueError("optimizer name not recognized")


def format_params(args):
    """Formats the command line arguments so that they can be logged.

    Args:
        The args returned from the `config` file.

    Returns:
        A formatted human readable string representation of the arguments.
    """
    formatted_args = ""
    for key in args.keys():
        formatted_params += "{} : {} \n".format(x, args[key])
        formatted_params += "\n"
    return formatted_params


def format_history(history, epochs):
    """Format the history into a readable string.

    Args:
        history: Object containing all the events that we are monitoring.
        epochs: The total number of epochs we are training for.

    Returns:
        A string that is human readable and can be used for logging.
    """
    monitored_val = history.keys()
    results = ""
    for epoch in range(epochs):
        res = "epoch {}/{} : ".format(epoch, epochs)
        for x in monitored_val:
            kv = " {} = {}".format(x, round(history[x][epoch], 3))
            res += ' | ' + kv
        res += "\n"
        results += res
    return results


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    existing_dirs = np.array([d for d in os.listdir(models_dir)
                             if os.path.isdir(os.path.join(models_dir,
                                                           d))]).astype(np.int)
    if len(existing_dirs) > 0:
        return os.path.join(models_dir, str(existing_dirs.max() + 1))
    else:
        return os.path.join(models_dir, '1')
