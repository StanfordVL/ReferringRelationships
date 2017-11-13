"""Utility functions for training.
"""

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.optimizers import RMSprop, Adam, Adagrad, Adadelta
from keras.callbacks import Callback
from keras import backend as K


import logging
import numpy as np
import os
import time


def weighted_cross_entropy(y_true, y_pred, w1, eps=10e-8):
    """Weighted cross entropy implementation.

    Args:
        y_true: Ground truth data.
        y_pred: Model predictions.
        w1: The weight of the positive ground truth values.
        eps: small epsilon value to avoid divide by zero errors.

    Returns:
        The loss value.
    """
    y_pred = K.clip(y_pred, eps, 1 - eps)
    loss_weights = 1. + (w1 - 1.) * y_true
    s_ce_values = - y_true * K.log(y_pred) - (1. - y_true) * K.log(1. - y_pred)
    loss = K.mean(s_ce_values * loss_weights)
    return loss


def get_loss_func(w1):
    """Wrapper for weighted sigmoid cross entropy loss.

    Args:
        w1: The weight for the positives.

    Return:
        The weighted sigmoid cross entropy loss function.
    """
    def loss_func(y_true, y_pred):
        return weighted_cross_entropy(y_true, y_pred, w1)
    return loss_func


def get_opt(opt, lr):
    """Initializes the opt that we want to train with.

    Args:
        opt: A string representation of the opt chosen by the user.
        lr: The learning rate used to initialize the optimizer.
        lr_decay: The learning rate decay used to initialize the optimizer.

    Returns:
        The keras optimizer we are going to use.
    """
    if opt=="rms":
        return RMSprop(lr=lr)
    elif opt=="adam":
        return Adam(lr=lr)
    elif opt=="adagrad":
        return Adagrad(lr=lr)
    elif opt=="adadelta":
        return Adadelta(lr=lr)
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


class LrReducer(Callback):
    """Lowers the learning rate when the val loss is not decreasing.
    """

    def __init__(self, args):
        """Constructor for the Logger.

        Args:
            args: Arguments passed in by the user.
        """
        super(Callback, self).__init__()
        self.patience = args.patience
        self.wait = 0
        self.best_loss = None
        self.lr_reduce_rate = args.lr_reduce_rate

    def on_epoch_end(self, epoch, logs={}):
        """Update the epoch count.

        Args:
            epoch: The epoch number we are on.
            logs: The training logs.
        """
        current_loss = logs.get('val_loss')
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            if self.wait >= self.patience:
                lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, lr*self.lr_reduce_rate)
            self.wait += 1


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
        res = "epoch %2d/%2d" % (self.epoch, self.total_epochs)
        for x in history.keys():
            kv = " %s: %3.3f" % (x, round(history[x], 3))
            res += ', ' + kv
        return res

    def on_train_begin(self, logs={}):
        """Function called before training starts.

        Args:
            logs: The training logs.
        """
        self.train_start_time = time.time()

    def on_train_end(self, logs={}):
        """Log the time it took to train.

        Args:
            logs: The training logs.
        """
        total_train_time = time.time() - self.train_start_time
        logging.info('Total training time for %d epochs was %2.3f' % (
            self.total_epochs, total_train_time))

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
            epoch: The epoch number we are on.
            logs: The training logs.
        """
        epoch_time = time.time() - self.epoch_start_time
        if self.log_every_batch:
            logging.info("="*30)
        logging.info("Epoch took %2.3f seconds" % epoch_time)
        lr = K.get_value(self.model.optimizer.lr)
        lr_string = 'lr: %0.5f, ' % lr
        logging.info(lr_string + self.format_logs(logs))
        if self.log_every_batch:
            logging.info("="*30)
        self.epoch += 1

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
            lr = K.get_value(self.model.optimizer.lr)
            lr_string = 'lr: %0.5f, ' % lr
            logging.info(time_string + lr_string + self.format_logs(logs))
