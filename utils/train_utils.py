"""Utility functions for training.
"""

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.optimizers import RMSprop, Adam, Adagrad, Adadelta
from keras.callbacks import Callback

import cv2
import logging
import numpy as np
import os
import time


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


def format_args(args):
    """Formats the command line arguments so that they can be logged.

    Args:
        The args returned from the `config` file.

    Returns:
        A formatted human readable string representation of the arguments.
    """
    formatted_args = "Training Arguments: \n"
    args = args.__dict__
    for key in args.keys():
        formatted_args += "\t > {} : {} \n".format(key, args[key])
    return formatted_args


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


class Logger(Callback):
    """A logging callback that tracks how well our model is doing over time.
    """

    def __init__(self, args):
        """Constructor for the Logger.

        Args:
            args: Arguments passed in by the user.
        """
        self.total_epochs = args.epochs
        self.log_every_batch = args.log_every_batch
        self.epoch = 1
        self.epoch_start_time = None
        self.batch_start_time = None

    def format_logs(self, history):
        """Format the history into a readable string.

        Args:
            history: Object containing all the events that we are monitoring.

        Returns:
            A string that is human readable and can be used for logging.
        """
        res = "epoch %2d/%2d : " % (self.epoch, self.total_epochs)
        for x in history.keys():
            kv = " %s = %3.3f" % (x, round(history[x], 3))
            res += ', ' + kv
        return res

    def on_train_end(self, logs={}):
        """Log the best validation loss at the end of training.

        Args:
            logs: The training logs.
        """
        total_train_time = time.time() - self.train_start_time
        logging.info('Total training time for %d epochs was %2.3f' % (
            self.total_epochs, total_train_time))
        logging.info('Best validation loss: {}'.format(
            round(np.min(logs['val_loss']), 4)))

    def on_epoch_begin(self, epoch, logs=None):
        """Record the time when it starts.

        Args:
            epoch: The epoch that is starting.
            logs: The training logs.
        """
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        """Update the epoch count.

        Args:
            logs: The training logs.
        """
        epoch_time = time.time() - self.epoch_start_time
        self.epoch += 1
        if self.log_every_batch:
            logging.info("="*30)
        logging.info("Epoch took %2.3f% seconds" % epoch_time)
        logging.info(self.format_logs(logs))
        if self.log_every_batch:
            logging.info("="*30)

    def on_batch_begin(self, batch, logs=None):
        """Record the time when it starts.

        Args:
            batch: The batch we are training on.
            logs: The training logs.
        """
        if self.log_every_batch:
            self.batch_start_time = time.time()

    def on_batch_end(self, batch, logs={}):
        """Log the progress of our training.

        Args:
            batch: The batch we are training on.
            logs: The training logs.
        """
        if self.log_every_batch:
            batch_time = time.time() - self.batch_start_time
            time_string = 'Time: %2.3f, ' % batch_time
            logging.info(time_string + self.format_logs(logs))
