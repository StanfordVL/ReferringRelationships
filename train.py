import numpy as np
import progressbar
import os
import cv2
import logging

from keras.optimizers import Adam
from config import *
from data import VRDDataset
from evaluation import *
from model import ReferringRelationshipsModel
from image_utils import visualize_weights

def get_subset(idx, data):
    res = []
    for x in data:
        res += [x[idx]]
    return res

res_dir = 'results/3'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(res_dir, 'train.log'))
logger.addHandler(fh)

# ******************************************* DATA *******************************************
data = VRDDataset()
logger.info("Building VRD dataset...")
subjects_data, relationships_data, objects_data, subjects_bbox, objects_bbox = data.build_dataset()
num_subjects = len(np.unique(subjects_data))
num_predicates = len(np.unique(relationships_data))
num_objects = len(np.unique(objects_data))
# image_data = data.get_images(image_ids)
N = subjects_data.shape[0]
permutation = np.arange(N)
np.random.shuffle(permutation)
train_idx = permutation[:int(N * (1 - validation_split))]
val_idx = permutation[int(N * (1 - validation_split)):]
# training data
train_subjects, train_predicates, train_objects, train_subject_bbox, train_object_bbox = get_subset(
        train_idx,
        [subjects_data, relationships_data, objects_data, subjects_bbox, objects_bbox])
N_train = len(train_idx)
# validation data
val_subjects, val_predicates, val_objects, val_subject_bbox, val_object_bbox = get_subset(
        val_idx, [subjects_data, relationships_data, objects_data, subjects_bbox, objects_bbox])
logger.info("Getting val images...")
val_images = data.get_images(val_idx)
N_val = len(val_idx)
# ************************************* OVERFIT 1 EXAMPLE *************************************
# N = 1
# k = 22
# image_ids = image_ids[k:k + 1]
# image_data = image_data[k:k + 1]
# subjects_data = subjects_data[k:k + 1]
# relationships_data = relationships_data[k:k + 1]
# objects_data = objects_data[k:k + 1]
# subjects_region_data = subjects_region_data[k:k + 1]
# objects_region_data = objects_region_data[k:k + 1]

# ***************************************** TRAINING *****************************************
best_o_iou = -1
best_s_iou = -1
relationships_model = ReferringRelationshipsModel(num_objects=num_objects, num_subjects=num_subjects, num_predicates=num_predicates)
model = relationships_model.build_model()
print(model.summary())
optimizer = Adam(lr=lr)
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=optimizer)
#k = np.random.choice(train_idx, 1)[-1]
#k = train_idx[-1]
#im_test, subj_test, obj_test = data.get_images_and_regions([k])
#cv2.imwrite(os.path.join(res_dir, 'original.png'), im_test[0])
#cv2.imwrite(os.path.join(res_dir, 'subject-gt.png'), 255*subj_test[0].reshape(input_dim, input_dim, 1))
for i in range(epochs):
    s_iou, o_iou = evaluate(model, val_images, val_subjects, val_predicates, val_objects, val_subject_bbox, val_object_bbox, iou_thresh, score_thresh, input_dim)
    if s_iou > best_s_iou:
        logger.info("saving best subject model...")
        model.save(os.path.join(res_dir, "best_subject_model.h5"))
        best_s_iou = s_iou
    if o_iou > best_o_iou:
        logger.info("saving best object model...")
        model.save(os.path.join(res_dir, "best_object_model.h5"))
        best_o_iou = o_iou
    s_loss_hist = []
    o_loss_hist = []
    logger.info("Epoch {}/{}".format(i+1, epochs))
    if (i + 1) % 15 == 0:
        lr /= 2.
        model.optimizer.lr.assign(lr)
    logger.info("learning rate: {}".format(lr))
    nb_steps = N_train / batch_size
    bar = progressbar.ProgressBar(maxval=nb_steps).start()
    for j in range(nb_steps):
        bar.update(j + 1)
        train_batch_image_idx = train_idx[j * batch_size:(j + 1) * batch_size]
        train_images, gt_subject_regions, gt_object_regions = data.get_images_and_regions(train_batch_image_idx)
        _, s_loss, o_loss = model.train_on_batch([train_images, train_subjects[j * batch_size:(j + 1) * batch_size],
                                                  train_predicates[j * batch_size:(j + 1) * batch_size],
                                                  train_objects[j * batch_size:(j + 1) * batch_size]],
                                                 [gt_subject_regions, gt_object_regions])
        s_loss_hist += [s_loss]
        o_loss_hist += [o_loss]
    bar.finish()
    logger.info("------------------------ subject loss: {}".format(np.mean(s_loss_hist)))
    logger.info("------------------------ object loss: {}".format(np.mean(o_loss_hist)))
    #subject_pred, object_pred = model.predict([im_test, subjects_data[k:k+1], relationships_data[k:k+1], objects_data[k:k+1]])
    #visualize_weights(im_test[0], subject_pred, input_dim, i, 'subject', res_dir)
