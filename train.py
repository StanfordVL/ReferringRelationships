import logging
import os
import cv2
import progressbar
from keras.optimizers import Adam

from config import *
from data import VRDDataset
from evaluation import *
from model import ReferringRelationshipsModel
from image_utils import visualize_weights

res_dir = 'results'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(res_dir, 'train.log'))
logger.addHandler(fh)

# ******************************************* DATA *******************************************
# creates dataset object that has json data path, image metadata etc ...
vrd_dataset = VRDDataset()
logger.info("Building VRD dataset...")
# get train image ids and validation image ids by using some split 
train_split, val_split = vrd_dataset.get_train_val_splits(train_val_split_ratio)
# get training data, each array has the same length 
# train_image_idx: array Nx1 with image ids, repeated for each relationship within an image 
# train_relationships: array Nx3 zith subject, predicate, object categories
# object and subject bbox: arrays Nx4 with top left bottom right coordinates for evaluation
train_image_idx, train_relationships, train_subject_bbox, train_object_bbox = vrd_dataset.build_dataset(train_split)
N_train = len(train_image_idx)
# todo : shuffle data 
logger.info("Number of training samples : {}".format(N_train))
logger.info("Getting val images...")
# doing the same for validation data 
val_image_idx, val_relationships, val_subject_bbox, val_object_bbox = vrd_dataset.build_dataset(val_split)
val_images = vrd_dataset.get_images(val_image_idx)
N_val = len(val_image_idx)
logger.info("Number of validation samples : {}".format(N_val))
# getting number of categories to build embedding layers in the model 
num_subjects = vrd_dataset.num_subjects
num_predicates = vrd_dataset.num_predicates
num_objects = vrd_dataset.num_objects

# ***************************************** TRAINING *****************************************
best_o_iou = -1
best_s_iou = -1
# build the model
relationships_model = ReferringRelationshipsModel(num_subjects=num_subjects, num_predicates=num_predicates,
                                                  num_objects=num_objects)
model = relationships_model.build_model()
print(model.summary())
optimizer = Adam(lr=lr)
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=optimizer)

for i in range(epochs):
    # predict object and subject regions to evaluate the current model 
    # these are numpy arrays 224 x 224 x 1
    s_regions_pred, o_regions_pred = model.predict(
            [val_images, val_relationships[:, 0], val_relationships[:, 1], val_relationships[:, 2]])
    # compute iou for each validation example 
    s_iou, o_iou = evaluate(s_regions_pred, o_regions_pred, train_subject_bbox, train_object_bbox, input_dim, score_thresh)
    s_iou_mean = s_iou.mean()
    o_iou_mean = o_iou.mean()
    logger.info("subject iou mean : {} \nsubject accuracy for iou thresh={} : {}".format(s_iou_mean, iou_thresh, (s_iou>iou_thresh).mean()))
    logger.info("object iou mean : {} \nobject accuracy for iou thresh={} : {}\n".format(o_iou_mean, iou_thresh, (o_iou>iou_thresh).mean()))
    if s_iou_mean > best_s_iou:
        logger.info("saving best subject model...")
        model.save(os.path.join(res_dir, "best_subject_model.h5"))
        best_s_iou = s_iou_mean
    if o_iou_mean > best_o_iou:
        logger.info("saving best object model...")
        model.save(os.path.join(res_dir, "best_object_model.h5"))
        best_o_iou = o_iou_mean
    s_loss_hist = []
    o_loss_hist = []
    logger.info("Epoch {}/{}".format(i + 1, epochs))
    if (i + 1) % step == 0:
        lr /= 2.
        model.optimizer.lr.assign(lr)
    logger.info("learning rate: {}".format(lr))
    # here I manually divide the training data in batches
    # this is the slow part 
    # first compute number of steps
    nb_steps = N_train / batch_size
    bar = progressbar.ProgressBar(maxval=nb_steps).start()
    for j in range(nb_steps):
        bar.update(j + 1)
        start, end = (j * batch_size, (j + 1) * batch_size)
        # get subset data for this batch 
        batch_image_idx = train_image_idx[start:end]
        batch_s_bbox = train_subject_bbox[start:end]
        batch_o_bbox = train_object_bbox[start:end]
        # call get_images_and_regions that returns 3 image arrays
        # batch_images: training images as numpy array
        # batch_s_regions: image for subject ground truth region 
        # batch_o_regions: image for object ground truth region
        batch_images, batch_s_regions, batch_o_regions = vrd_dataset.get_images_and_regions(batch_image_idx,
                                                                                            batch_s_bbox, batch_o_bbox)
        _, s_loss, o_loss = model.train_on_batch(
                [batch_images, train_relationships[start:end, 0],
                 train_relationships[start:end, 1],
                 train_relationships[start:end, 2]],
                [batch_s_regions, batch_o_regions])
        s_loss_hist += [s_loss]
        o_loss_hist += [o_loss]
    bar.finish()
    logger.info("------------------------ subject loss: {}".format(np.mean(s_loss_hist)))
    logger.info("------------------------ object loss: {}".format(np.mean(o_loss_hist)))
