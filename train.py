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
vrd_dataset = VRDDataset()
logger.info("Building VRD dataset...")
train_split, val_split = vrd_dataset.get_train_val_splits(train_val_split_ratio)
train_image_idx, train_relationships, train_subject_bbox, train_object_bbox = vrd_dataset.build_dataset(train_split)
N_train = len(train_image_idx)
logger.info("Number of training samples : {}".format(N_train))
logger.info("Getting val images...")
val_image_idx, val_relationships, val_subject_bbox, val_object_bbox = vrd_dataset.build_dataset(val_split)
val_images = vrd_dataset.get_images(val_image_idx)
N_val = len(val_image_idx)
logger.info("Number of validation samples : {}".format(N_val))
num_subjects = vrd_dataset.num_subjects
num_predicates = vrd_dataset.num_predicates
num_objects = vrd_dataset.num_objects

# ***************************************** TRAINING *****************************************
best_o_iou = -1
best_s_iou = -1
relationships_model = ReferringRelationshipsModel(num_subjects=num_subjects, num_predicates=num_predicates,
                                                  num_objects=num_objects)
model = relationships_model.build_model()
print(model.summary())
optimizer = Adam(lr=lr)
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=optimizer)

# overfitting one example 
train_images, train_s_regions, train_o_regions = vrd_dataset.get_images_and_regions(train_image_idx, train_subject_bbox, train_object_bbox)
k = np.random.choice(np.arange(len(train_image_idx)), 1)[0]
cv2.imwrite(os.path.join(res_dir, 'original.png'), train_images[k])
cv2.imwrite(os.path.join(res_dir, 'subject-gt.png'), 255*train_s_regions[k].reshape(input_dim, input_dim, 1))
for i in range(epochs):
    #s_regions_pred, o_regions_pred = model.predict(
    #        [val_images, val_relationships[:, 0], val_relationships[:, 1], val_relationships[:, 2]])
    #s_iou, o_iou = evaluate(s_regions_pred, o_regions_pred, val_subject_bbox, val_object_bbox, input_dim, score_thresh)
    s_regions_pred, o_regions_pred = model.predict([train_images, train_relationships[:, 0], train_relationships[:, 1], train_relationships[:, 2]])
    visualize_weights(train_images[k], s_regions_pred[k], input_dim, i, 'subject', res_dir)
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
    nb_steps = N_train / batch_size
    bar = progressbar.ProgressBar(maxval=nb_steps).start()
    for j in range(nb_steps):
        bar.update(j + 1)
        start, end = (j * batch_size, (j + 1) * batch_size)
        batch_image_idx = train_image_idx[start:end]
        batch_s_bbox = train_subject_bbox[start:end]
        batch_o_bbox = train_object_bbox[start:end]
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
    # subject_pred, object_pred = model.predict([im_test, subjects_data[k:k+1], relationships_data[k:k+1], objects_data[k:k+1]])
    # visualize_weights(im_test[0], subject_pred, input_dim, i, 'subject', res_dir)




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
