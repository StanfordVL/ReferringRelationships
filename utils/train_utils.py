"""Utility functions for training.
"""

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.optimizers import RMSprop, Adam, Adagrad, Adadelta
from keras.callbacks import Callback

import logging
import numpy as np
import os
import time


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
                lr = self.model.optimizer.lr.get_value()
                self.model.optimizer.lr.set_value(lr*self.lr_reduce_rate)
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
        logging.info(self.format_logs(logs))
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
            logging.info(time_string + self.format_logs(logs))
