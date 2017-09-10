import json
import logging
import os
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from ReferringRelationships.config import params
from ReferringRelationships.iterator import RefRelDataIterator
from ReferringRelationships.model import ReferringRelationshipsModel
from ReferringRelationships.utils import format_params, get_dir_name, format_history


if not params["session_params"]["save_dir"]:
    params["session_params"]["save_dir"] = get_dir_name(params["session_params"]["models_dir"])
    os.makedirs(params["session_params"]["save_dir"])

json.dump(params, open(os.path.join(params["session_params"]["save_dir"], "params.json"), "w"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(params["session_params"]["save_dir"], 'train.log'))
logger.addHandler(fh)
logger.info(format_params(params))


def iou(y_true, y_pred):
    # todo: check this
    #import ipdb;
    #ipdb.set_trace()
    input_dim = params["model_params"]["input_dim"]
    y_true = tf.reshape(y_true, [-1, input_dim * input_dim])
    y_pred = tf.reshape(y_pred, [-1, input_dim * input_dim])
    y_pred = tf.cast(y_pred > params["eval_params"]["score_thresh"], tf.float32)
    intersection = tf.cast(y_true * y_pred > 0, tf.float32)
    union = tf.cast(y_true + y_pred, tf.float32)
    iou_values = K.sum(intersection, axis=-1) / K.sum(union, axis=-1)
    return K.mean(iou_values)


def binary_ce(y_true, y_pred):
    # todo: check how to compuye ce with multidimensinal tensors
    input_dim = params["model_params"]["input_dim"]
    ce = K.binary_crossentropy(y_true, y_pred)
    ce = tf.reshape(ce, [-1, input_dim * input_dim])
    return K.mean(ce, axis=-1)


# ******************************************* DATA *******************************************
train_generator = RefRelDataIterator(params["data_params"]["image_data_dir"], params["data_params"]["train_data_dir"], input_dim=params["model_params"]["input_dim"], batch_size=params["session_params"]["batch_size"])
val_generator = RefRelDataIterator(params["data_params"]["image_data_dir"], params["data_params"]["val_data_dir"], input_dim=params["model_params"]["input_dim"], batch_size=params["session_params"]["batch_size"])
logger.info("Train on {} samples".format(train_generator.samples))
logger.info("Validate on {} samples".format(val_generator.samples))

# ***************************************** TRAINING *****************************************
relationships_model = ReferringRelationshipsModel(params["model_params"])
model = relationships_model.build_model()
model.summary(print_fn=lambda x: logger.info(x + "\n"))
optimizer = Adam(lr=params["session_params"]["lr"])
#model.compile(loss=[binary_ce, binary_ce], optimizer=optimizer, metrics=[iou, iou])
# TODO: fix iou
model.compile(loss=["binary_crossentropy", "binary_crossentropy"], optimizer=optimizer, metrics=["acc", "acc"])
checkpointer = ModelCheckpoint(
    filepath=os.path.join(params["session_params"]["save_dir"], "model{epoch:02d}-{val_loss:.2f}.h5"), verbose=1,
    save_best_only=False)
history = model.fit_generator(train_generator, steps_per_epoch=int(train_generator.samples/params["session_params"]["batch_size"]), epochs=params["session_params"]["epochs"], validation_data=val_generator,
                    validation_steps=int(val_generator.samples/params["session_params"]["batch_size"]), callbacks=[checkpointer]).history
logger.info(format_history(history, params["session_params"]["epochs"]))
# logger.info("Best validation accuracy : {}".format(round(np.max(hist['val_acc']), 4)))
