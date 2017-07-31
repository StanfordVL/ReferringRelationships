from data import VisualGenomeRelationshipsDataset
from config import *


from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, UpSampling2D, Reshape, Input, Lambda, merge
from keras.layers.merge import Dot
from keras.optimizers import RMSprop
import numpy as np


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
def relationship_model(num_triplets, embedding_dim, hidden_dim):
    relationship_branch = Sequential()
    relationship_branch.add(Dense(embedding_dim, input_shape=(num_triplets,)))
    relationship_branch.add(Dense(hidden_dim, activation='relu'))
    relationship_branch.add(Dropout(0.2))
    # relationship_branch.add(RepeatVector(feat_map_dim * feat_map_dim))  # (14x14)x100
    return relationship_branch


# ******************************************* DATA *******************************************
# x_im = 255 * np.random.random((3, 224, 224, 3))  # 3 random images
# x_rel = np.array([4, 5, 1])  # 3 relationships in range(num_triplets)
# x_rel = keras.utils.to_categorical(x_rel, num_triplets)
# y_regions = np.zeros((3, 224, 224, 1))  # label regions, check how to reshape
# y_regions = np.zeros((3, 224 * 224))  # label regions, check how to reshape
# y_regions = flatten_model(input_dim)(y_regions)
data = VisualGenomeRelationshipsDataset(data_path="data/subset_1/subset_relationships.json")
image_ids, relationships, gt_regions = data.build_dataset()
image_data = data.get_images(image_ids)
relationship_data = to_categorical(relationships, num_triplets)
N = gt_regions.shape[0]
# image_ids (N,)
# relationships (N,)
# gt_regions (N, im_dim, im_dim)
# images (N, im_dim, im_dim, 3)


# ****************************************** MODEL ******************************************
input_im = Input(shape=(input_dim, input_dim, 3))
input_rel = Input(shape=(num_triplets,))
images = image_model(input_dim, feat_map_dim, hidden_dim)(input_im)
images = Reshape(target_shape=(feat_map_dim * feat_map_dim, hidden_dim))(images)  # (196)x100
relationships = relationship_model(num_triplets, embedding_dim, hidden_dim)(input_rel)
relationships = Reshape(target_shape=(1, hidden_dim))(relationships)  # (196)x100
# todo: try to avoid reshaping by adding a custom dot layer that takes multi dimensional inputs
# distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([images, relationships])
# merged = merge([images, relationships], mode='dot', dot_axes=(2, 2))
merged = Dot(axes=(2, 2))([images, relationships])
reshaped = Reshape(target_shape=(feat_map_dim, feat_map_dim, 1))(merged)
upsampled = UpSampling2D(size=(upsampling_factor, upsampling_factor))(reshaped)
flattened = Flatten(input_shape=(input_dim, input_dim, 10))(upsampled)
model = Model(inputs=[input_im, input_rel], outputs=[flattened])
rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

# ***************************************** TRAINING *****************************************
# todo: check flatten and reshaping are the same in numpy and keras
history = model.fit([image_data, relationship_data], gt_regions.reshape(N, -1), batch_size=batch_size, epochs=epochs, verbose=1)
