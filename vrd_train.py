import logging
import os
import cv2
import progressbar
from keras.optimizers import Adam
from keras import backend as K 
from config import *
from data import VRDDataset
from evaluation import *
from model import ReferringRelationshipsModel
from image_utils import visualize_weights
from iterator import RefRelDataIterator

res_dir = 'results'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(res_dir, 'train.log'))
logger.addHandler(fh)

# ******************************************* DATA *******************************************
num_subjects=100
num_predicates=70
num_objects=100
image_dir = "/data/chami/VRD/sg_dataset/sg_train_images/" 
train_data_dir = "/data/chami/VRD/train/"
val_data_dir = "/data/chami/VRD/val/"
train_generator = RefRelDataIterator(image_dir, train_data_dir)
val_generator = RefRelDataIterator(image_dir, val_data_dir)

def binary_ce(y_true, y_pred):
    # todo: check how to compuye ce with multidimensinal tensors 
    n = y_true.shape[0]
    return K.mean(K.binary_crossentropy(y_true.reshape(n, -1), y_pred.reshape(n, -1)), axis=-1)
    

# ***************************************** TRAINING *****************************************
# build the model
relationships_model = ReferringRelationshipsModel(num_subjects=num_subjects, num_predicates=num_predicates, num_objects=num_objects)
model = relationships_model.build_model()
print(model.summary())
optimizer = Adam(lr=lr)
# model.compile(loss=[binary_ce, binary_ce], optimizer=optimizer, metrics=[subject_iou, object_iou])
model.compile(loss=["binary_crossentropy", "binary_crossentropy"], optimizer=optimizer)
#for epoch in range(epochs):
#    print("Epoch : {}/{}".format(epoch, epochs))
model.fit_generator(train_generator, steps_per_epoch=10, epochs=1, validation_data=val_generator, validation_steps=10)
    #pred = model.predict(pdf_image)[0]
    #save_attention_weights(pdf_image, pred, os.path.join(res_dir, 'attention-{}.png'.format(epoch)), img_height, img_width)

