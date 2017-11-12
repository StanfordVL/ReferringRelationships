"""Training script for Segmentation.
"""

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.metrics import categorical_accuracy

from config import parse_args
from models import SemanticSegmentationModel
from iterator import SemanticSegmentationIterator
from eval_utils import format_results
from eval_utils import pixel_acc, mean_iu, sparse_accuracy_ignoring_first_label
from train_utils import Logger
from train_utils import LrReducer
from train_utils import get_loss_func, multinomial_logistic_loss, softmax_sparse_crossentropy_ignoring_first_label
from train_utils import get_dir_name
from train_utils import get_opt
from train_utils import format_args
from fcnn import AtrousFCN_Resnet50_16s

import json
import logging
import numpy as np
import os


if __name__=='__main__':
    # Parse command line arguments.
    args = parse_args()

    # First check if --use-models-dir is set. In that case, create a new folder
    # to store all the training logs.
    if (args.use_models_dir and
        args.models_dir is not None and
        args.save_dir is None):
        args.save_dir = get_dir_name(args.models_dir)

    # If the save directory does exists, don't launch the training script.
    if not args.overwrite and os.path.isdir(args.save_dir):
        raise ValueError('The directory %s already exists. Exiting training!'
              % args.save_dir)

    # Otherwise, create the directory and start training.
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    # Setup logging.
    logfile = os.path.join(args.save_dir, 'train.log')
    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=logfile)

    # Store the arguments used in this training process.
    args_file = open(os.path.join(args.save_dir, 'args.json'), 'w')
    json.dump(args.__dict__, args_file)
    args_file.close()
    logging.info(format_args(args))

    # Choose the correct Iterator and Model.
    #SegmentationModel = SemanticSegmentationModel
    Iterator = SemanticSegmentationIterator
    #if args.task == 'class':
    #    SegmentationModel = ClassSegmentationModel
    #    Iterator = ClassSegmentationIterator

    # Setup the training and validation data iterators
    train_generator = Iterator(args.train_data_dir, args)
    val_generator = Iterator(args.val_data_dir, args)
    logging.info('Train on {} samples'.format(train_generator.samples))
    logging.info('Validate on {} samples'.format(val_generator.samples))

    # Setup all the metrics we want to report. The names of the metrics need to
    # be set so that Keras can log them correctly.
    #metrics = [mean_iu, categorical_accuracy, sparse_accuracy_ignoring_first_label]
    metrics = [sparse_accuracy_ignoring_first_label]
    # create a new instance model
    #segmentation_model = SegmentationModel(args)
    model = AtrousFCN_Resnet50_16s((224,224,3), weight_decay=0.0001/2, batch_momentum=0.95, classes=args.num_objects)
    model.summary(print_fn=lambda x: logging.info(x + '\n'))
    #optimizer = get_opt(opt=args.opt, lr=args.lr)
    lr_base = 0.01 * (float(args.batch_size) / 16)
    optimizer = SGD(lr=lr_base, momentum=0.9)
    # get the loss function and compile the model
    #if args.loss_func == 'weighted':
    #    loss_func = get_loss_func(args.w1)
    #else:
    #    loss_func = multinomial_logistic_loss
    losses = softmax_sparse_crossentropy_ignoring_first_label
    model.compile(loss=losses, optimizer=optimizer, metrics=metrics)

    # load model weights from checkpoint
    if args.model_checkpoint:
         model.load_weights(args.model_checkpoint)

    # Setup callbacks for tensorboard, logging and checkpoints.
    tb_callback = TensorBoard(log_dir=args.save_dir)
    lr_reducer_callback = LrReducer(args)
    logging_callback = Logger(args)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(
            args.save_dir, 'model{epoch:02d}-{val_loss:.2f}.h5'),
        verbose=1,
        save_weights_only=True,
        save_best_only=args.save_best_only,
        monitor='val_loss')

    # Start training.
    if args.train_steps_per_epoch < 0:
        train_steps = len(train_generator)
    else:
        train_steps = args.train_steps_per_epoch
    if args.val_steps_per_epoch < 0:
        val_steps = len(val_generator)
    else:
        val_steps = args.val_steps_per_epoch
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps,
                        epochs=args.epochs,
                        validation_data=val_generator,
                        validation_steps=val_steps,
                        verbose=2,
                        use_multiprocessing=args.multiprocessing,
                        workers=args.workers,
                        shuffle=args.shuffle,
                        callbacks=[checkpointer, tb_callback, logging_callback,
                                   lr_reducer_callback])

    # Run Evaluation.
    val_steps = len(val_generator)
    outputs = model.evaluate_generator(generator=val_generator,
                                       steps=val_steps,
                                       use_multiprocessing=args.multiprocessing,
                                       workers=args.workers)
    logging.info('='*30)
    logging.info('Evaluation results - ' + format_results(model.metrics_names,
                                                          outputs))


    # Run Testing.
    test_generator = Iterator(args.test_data_dir, args)
    test_steps = len(test_generator)
    outputs = model.evaluate_generator(generator=test_generator,
                                       steps=test_steps,
                                       use_multiprocessing=args.multiprocessing,
                                       workers=args.workers)
    logging.info('='*30)
    logging.info('Test results - ' + format_results(model.metrics_names,
                                                    outputs))
