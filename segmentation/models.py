"""Segmentation model.
"""

from config import parse_args
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, UpSampling2D, Input, Activation, BatchNormalization, RepeatVector
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.core import Lambda, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Multiply, Dot, Add, Concatenate
from keras.models import Model

import numpy as np

class BaseModel(object):
    """Contains helper functions.
    """

    def __init__(self, args):
        """Constructor for Semantic segmentation.

        Args:
            args: The arguments specified by `config.py`
        """
        self.input_dim = args.input_dim
        self.feat_map_dim = args.feat_map_dim
        self.hidden_dim = args.hidden_dim
        self.num_objects = args.num_objects
        self.dropout = args.dropout
        self.num_objects = args.num_objects
        self.feat_map_layer = args.feat_map_layer

    def get_image_features(self, input_im):
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(self.input_dim, self.input_dim, 3))
        for layer in base_model.layers:
            layer.trainable = False
        output = base_model.get_layer(self.feat_map_layer).output
        image_branch = Model(inputs=base_model.input, outputs=output)
        im_features = image_branch(input_im)
        im_features = Dropout(self.dropout)(im_features)
        im_features = Conv2D(self.hidden_dim, 1,
                             strides=(1, 1), padding='same',
                             activation='relu')(im_features)
        im_features = Dropout(self.dropout)(im_features)
        return im_features

    def attend(self, feature_map, query):
        query = Reshape((1, 1, self.hidden_dim,))(query)
        attention_weights = Multiply()([feature_map, query])
        attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(attention_weights)
        attention_weights = Activation("relu")(attention_weights)
        return attention_weights

    def upsample(self, res):
        upsampling_factor = self.input_dim / self.feat_map_dim
        k = int(np.log(upsampling_factor) / np.log(2))
        for i in range(k):
            res = UpSampling2D(size=(2, 2))(res)
            res = Conv2DTranspose(1, 3, padding='same', use_bias=False, activation="relu")(res)
        return res


class SemanticSegmentationModel(BaseModel):
    """Semantic segmentation model give the category.
    """

    def build_model(self):
        input_image = Input(shape=(self.input_dim, self.input_dim, 3))
        image_features = self.get_image_features(input_image)
        # adding 1 for background class
        object_regions = [Dense(1, activation="relu")(image_features) for i in range(self.num_objects)]
        upsampled_regions = Concatenate(axis=3)([self.upsample(x) for x in object_regions])
        output = Activation('softmax')(upsampled_regions)
        model = Model(inputs=input_image, outputs=output)
        return model


class ClassSegmentationModel(BaseModel):
    """Semantic segmentation model give the category.
    """

    def build_model(self):
        input_image = Input(shape=(self.input_dim, self.input_dim, 3))
        input_object = Input(shape=(1,))
        image_features = self.get_image_features(input_image)
        embed = Embedding(self.num_objects, self.hidden_dim, input_length=1)
        embedded_object = embed(input_object)
        embedded_object = Activation("relu")(embedded_object)
        object_att = self.attend(image_features, embedded_object)
        object_regions = self.upsample(object_att)
        model = Model(inputs=[input_image, input_object], outputs=object_regions)
        return model

if __name__ == "__main__":
    args = parse_args()
    segm = SemanticSegmentationModel(args)
    #segm = ClassSegmentationModel(args)
    model = segm.build_model()
    print(model.summary())
