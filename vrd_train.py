import logging
import os
import cv2
import progressbar
from keras.optimizers import Adam
from keras import backend as K 
from config import *
from data import VRDDataset
from evaluation import iou
from model import ReferringRelationshipsModel
from image_utils import visualize_weights
from iterator import RefRelDataIterator
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

res_dir = 'results'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(res_dir, 'train.log'))
logger.addHandler(fh)

def binary_ce(y_true, y_pred):
    # todo: check how to compuye ce with multidimensinal tensors 
    ce = K.binary_crossentropy(y_true, y_pred)
    ce = tf.reshape(ce, [-1, input_dim*input_dim])
    return K.mean(ce, axis=-1)


# ******************************************* DATA *******************************************
image_dir = "/data/chami/VRD/sg_dataset/sg_train_images/" 
train_data_dir = "/data/chami/VRD/train/"
val_data_dir = "/data/chami/VRD/val/"
train_generator = RefRelDataIterator(image_dir, train_data_dir)
val_generator = RefRelDataIterator(image_dir, val_data_dir)
    

# ***************************************** TRAINING *****************************************
# build the model
relationships_model = ReferringRelationshipsModel(num_subjects=num_subjects, num_predicates=num_predicates, num_objects=num_objects)
model = relationships_model.build_model()
print(model.summary())
optimizer = Adam(lr=lr)
model.compile(loss=[binary_ce, binary_ce], optimizer=optimizer,  metrics=[iou, iou])
checkpointer = ModelCheckpoint(filepath=os.path.join(res_dir, "model.h5"), verbose=1, save_best_only=True)
model.fit_generator(train_generator, steps_per_epoch=10, epochs=10, validation_data=train_generator, validation_steps=10, callbacks=[checkpointer])
