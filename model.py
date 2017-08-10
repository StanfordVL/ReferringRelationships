from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, UpSampling2D, Reshape, Input
from keras.layers.merge import Dot, Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from config import *
from data import VisualGenomeRelationshipsDataset
from image_utils import save_predictions

import numpy as np
import cv2
import os
# ******************************************* DATA *******************************************
data = VisualGenomeRelationshipsDataset(data_path="data/VisualGenome/subset_1/subset_relationships.json")
image_ids, subjects_data, relationships_data, objects_data, gt_regions = data.build_dataset()
num_subjects = len(np.unique(subjects_data))
num_predicates = len(np.unique(relationships_data))
num_objects = len(np.unique(objects_data))
image_data = data.get_images(image_ids)
#relationship_data = to_categorical(relationships, num_triplets)
N = gt_regions.shape[0]
# image_ids (N,)
# relationships (N,)
# gt_regions (N, im_dim, im_dim)
# images (N, im_dim, im_dim, 3)

# ************************************* OVERFIT 1 EXAMPLE *************************************
N = 1
k = 9
image_ids = image_ids[k:k+1]
image_data = image_data[k:k+1]
subjects_data = subjects_data[k:k+1]
relationships_data = relationships_data[k:k+1]
objects_data = objects_data[k:k+1]
gt_regions = gt_regions[k:k+1]

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
    #relationship_branch = Sequential()
    rel_repr = Dense(hidden_dim, activation='relu')(concatenated_inputs)
    #rel_repr = Dropout(0.2)(rel_repr)
    return rel_repr

# ****************************************** MODEL ******************************************
input_im = Input(shape=(input_dim, input_dim, 3))
input_rel = Input(shape=(1,))
input_obj = Input(shape=(1,))
input_subj = Input(shape=(1,))
images = image_model(input_dim, feat_map_dim, hidden_dim)(input_im)
images = Reshape(target_shape=(feat_map_dim * feat_map_dim, hidden_dim))(images)  # (196)x100
relationships = relationship_model(embedding_dim, hidden_dim, num_subjects, num_predicates, num_objects, input_subj, input_rel, input_obj)
#relationships = Reshape(target_shape=(1, hidden_dim))(relationships)  # (196)x100
# todo: try to avoid reshaping by adding a custom dot layer that takes multi dimensional inputs
# distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([images, relationships])
# merged = merge([images, relationships], mode='dot', dot_axes=(2, 2))
merged = Dot(axes=(2, 2))([images, relationships])
reshaped = Reshape(target_shape=(feat_map_dim, feat_map_dim, 1))(merged)
upsampled = UpSampling2D(size=(upsampling_factor, upsampling_factor))(reshaped)
flattened = Flatten(input_shape=(input_dim, input_dim, 10))(upsampled)
model = Model(inputs=[input_im, input_subj, input_rel, input_obj], outputs=[flattened])
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam)

# ***************************************** TRAINING *****************************************
# todo: check flatten and reshaping are the same in numpy and keras
cv2.imwrite(os.path.join('results/2', 'original.png'), image_data[0])
for i in range(epochs):
    print("Epoch {}/{}".format(i, epochs))
    history = model.fit([image_data, subjects_data, relationships_data, objects_data], gt_regions.reshape(N, -1), batch_size=batch_size, epochs=1, verbose=1)
    predictions = model.predict([image_data, subjects_data, relationships_data, objects_data])
    predictions = predictions.reshape(input_dim, input_dim, 1)
    cv2.imwrite(os.path.join('results/2', 'attention-' + str(i) + '.png'), predictions + image_data[0])
