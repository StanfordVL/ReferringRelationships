import os

import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, UpSampling2D, Reshape, Input, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import Dot, Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam

from config import *
from data import VisualGenomeRelationshipsDataset, VRDDataset
from evaluation import *

def get_subset(idx, data):
    res = []
    for x in data:
        res += [x[idx]]
    return res

# ******************************************* DATA *******************************************
data = VRDDataset()
print("Building VRD dataset...")
image_ids, subjects_data, relationships_data, objects_data, subjects_region_data, objects_region_data, subjects_bbox, objects_bbox  = data.build_dataset()
num_subjects = len(np.unique(subjects_data))
num_predicates = len(np.unique(relationships_data))
num_objects = len(np.unique(objects_data))
#image_data = data.get_images(image_ids)
N = objects_region_data.shape[0]
permutation = np.arange(N)
np.random.shuffle(permutation)
train_idx = permutation[:int(N*(1-validation_split))]
val_idx = permutation[int(N*(1-validation_split)):]
# training data
train_image_idx, train_subjects, train_predicates, train_objects, train_subject_regions, train_object_regions, train_subject_bbox, train_object_bbox = get_subset(train_idx, [image_ids, subjects_data, relationships_data, objects_data, subjects_region_data, objects_region_data, subjects_bbox, objects_bbox])
#print("Getting train images...")
#train_images = data.get_images(train_image_idx)
N_train = len(train_idx)
# validation data
val_image_idx, val_subjects, val_predicates, val_objects, val_subject_regions, val_object_regions, val_subject_bbox, val_object_bbox = get_subset(val_idx, [image_ids, subjects_data, relationships_data, objects_data, subjects_region_data, objects_region_data, subjects_bbox, objects_bbox])
print("Getting val images...")
val_images = data.get_images(val_image_idx)
N_val = len(val_idx)
# ************************************* OVERFIT 1 EXAMPLE *************************************
#N = 1
#k = 22
#image_ids = image_ids[k:k + 1]
#image_data = image_data[k:k + 1]
#subjects_data = subjects_data[k:k + 1]
#relationships_data = relationships_data[k:k + 1]
#objects_data = objects_data[k:k + 1]
#subjects_region_data = subjects_region_data[k:k + 1]
#objects_region_data = objects_region_data[k:k + 1]


# *************************************** FLATTEN MODEL ***************************************
# this is to make sure we preserve the ordering
def flatten_model(input_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(input_dim, input_dim, 1)))
    return model


# *************************************** IMAGE BRANCH ***************************************
def image_model(input_dim, feat_map_dim, hidden_dim):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_dim, input_dim, 3))
    for layer in base_model.layers:
        layer.trainable = False
    output = base_model.get_layer('block5_conv3').output
    output = Dense(hidden_dim)(output)
    image_branch = Model(inputs=base_model.input, outputs=output)
    return image_branch


# ************************************ RELATIONSHIP BRANCH ***********************************
def embedding(input_vector, num_categories, embedding_dim):
    embedding = Embedding(num_categories, embedding_dim, input_length=1)(input_vector)
    return embedding


def relationship_model(embedding_dim, hidden_dim, num_subjects, num_predicates, num_objects, input_subj, input_rel, input_obj):
    subj_embedding = embedding(input_subj, num_subjects, embedding_dim)
    predicate_embedding = embedding(input_rel, num_predicates, embedding_dim)
    obj_embedding = embedding(input_obj, num_objects, embedding_dim)
    concatenated_inputs = Concatenate(axis=2)([subj_embedding, predicate_embedding, obj_embedding])
    # relationship_branch = Sequential()
    rel_repr = Dense(hidden_dim, activation='relu')(concatenated_inputs)
    # rel_repr = Dropout(0.2)(rel_repr)
    return rel_repr


def attention_layer(image_features, relationships_features):
    merged = Dot(axes=(2, 2))([images, relationships])
    reshaped = Reshape(target_shape=(feat_map_dim, feat_map_dim, 1))(merged)
    upsampled = UpSampling2D(size=(upsampling_factor, upsampling_factor))(reshaped)
    flattened = Flatten(input_shape=(input_dim, input_dim, 10))(upsampled)
    predictions = Activation('sigmoid')(flattened)
    return predictions

# ****************************************** MODEL ******************************************
input_im = Input(shape=(input_dim, input_dim, 3))
input_rel = Input(shape=(1,))
input_obj = Input(shape=(1,))
input_subj = Input(shape=(1,))
images = image_model(input_dim, feat_map_dim, hidden_dim)(input_im)
images = Reshape(target_shape=(feat_map_dim * feat_map_dim, hidden_dim))(images)  # (196)x100
relationships = relationship_model(embedding_dim, hidden_dim, num_subjects, num_predicates, num_objects, input_subj, input_rel, input_obj)
# relationships = Reshape(target_shape=(1, hidden_dim))(relationships)  # (196)x100
# todo: try to avoid reshaping by adding a custom dot layer that takes multi dimensional inputs
# distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([images, relationships])
# merged = merge([images, relationships], mode='dot', dot_axes=(2, 2))
# merged = Dot(axes=(2, 2))([images, relationships])
# reshaped = Reshape(target_shape=(feat_map_dim, feat_map_dim, 1))(merged)
# upsampled = UpSampling2D(size=(upsampling_factor, upsampling_factor))(reshaped)
# flattened = Flatten(input_shape=(input_dim, input_dim, 10))(upsampled)
subject_regions = attention_layer(images, relationships)
object_regions = attention_layer(images, relationships)
model = Model(inputs=[input_im, input_subj, input_rel, input_obj], outputs=[subject_regions, object_regions])
adam = Adam(lr=lr)
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=adam)


def evaluate(model, val_images, val_subjects, val_predicates, val_objects, val_subject_bbox, val_object_bbox, iou_thresh, score_thresh):
    s_iou = []
    o_iou = []
    subject_preds, object_preds = model.predict([val_images, val_subjects, val_predicates, val_objects])
    for i in range(len(subject_preds)):
        pred_subject_bbox = get_bbox_from_heatmap(subject_preds[i].reshape(input_dim, input_dim), score_thresh)
        pred_object_bbox = get_bbox_from_heatmap(object_preds[i].reshape(input_dim, input_dim), score_thresh)
        s_iou += [compute_iou(pred_subject_bbox, val_subject_bbox[i])]
        o_iou += [compute_iou(pred_object_bbox, val_object_bbox[i])]
    s_iou = np.array(s_iou)
    o_iou = np.array(o_iou)
    return s_iou.mean(), (s_iou>iou_thresh).mean(), o_iou.mean(), (o_iou>iou_thresh).mean()
    

# ***************************************** TRAINING *****************************************
# todo: check flatten and reshaping are the same in numpy and keras
k = 22
#cv2.imwrite(os.path.join('results/2', 'original.png'), train_images[k])
#cv2.imwrite(os.path.join('results/2', 'gt.png'), 255*train_subject_regions[k])
for i in range(epochs):
    s_loss_hist = []
    o_loss_hist = []
    print("Epoch {}/{}".format(i, epochs))
    if (i+1)%20==0:
        lr /= 2.
        model.optimizer.lr.assign(lr)
    print("learning rate: {}".format(lr))
    for j in range(N_train/batch_size):
        train_batch_image_idx = train_image_idx[j*batch_size:(j+1)*batch_size]
        train_images = data.get_images(train_batch_image_idx)
        #history = model.fit([train_images, train_subjects[j*batch_size:(j+1)*batch_size], train_predicates[j*batch_size:(j+1)*batch_size], train_objects[j*batch_size:(j+1)*batch_size]], [train_subject_regions[j*batch_size:(j+1)*batch_size].reshape(batch_size, -1), train_object_regions[j*batch_size:(j+1)*batch_size].reshape(batch_size, -1)], batch_size=batch_size, epochs=1, verbose=1)
        _, s_loss, o_loss = model.train_on_batch([train_images, train_subjects[j*batch_size:(j+1)*batch_size], train_predicates[j*batch_size:(j+1)*batch_size], train_objects[j*batch_size:(j+1)*batch_size]], [train_subject_regions[j*batch_size:(j+1)*batch_size].reshape(batch_size, -1), train_object_regions[j*batch_size:(j+1)*batch_size].reshape(batch_size, -1)]) 
        s_loss_hist += [s_loss]
        o_loss_hist += [o_loss]
    print("subject loss: {}".format(np.mean(s_loss_hist)))
    print("object loss: {}".format(np.mean(o_loss_hist)))
    # saving attention heatmaps for one example 
    #subject_pred, object_pred = model.predict([train_images[k:k+1], train_subjects[k:k+1], train_predicates[k:k+1], train_objects[k:k+1]])
    #subject_pred = subject_pred.reshape(input_dim, input_dim, 1)
    #image_pred = np.zeros((input_dim, input_dim, 3))
    #image_pred += subject_pred*255
    #cv2.imwrite(os.path.join('results/2', 'attention-' + str(i) + '.png'), cv2.addWeighted(train_images[k], 0.6, image_pred, 0.4, 0))
    #s_iou_mean, s_iou_acc, o_iou_mean, o_iou_acc = evaluate(model, train_images, train_subjects, train_predicates, train_objects, train_subject_bbox, train_object_bbox, iou_thresh, score_thresh)
    s_iou_mean, s_iou_acc, o_iou_mean, o_iou_acc = evaluate(model, val_images, val_subjects, val_predicates, val_objects, val_subject_bbox, val_object_bbox, iou_thresh, score_thresh)
    print("subject iou mean : {} \n subject accuracy for iou thresh={} : {}".format(s_iou_mean, iou_thresh, s_iou_acc))
    print("object iou mean : {} \n object accuracy for iou thresh={} : {}".format(o_iou_mean, iou_thresh, o_iou_acc))
    
