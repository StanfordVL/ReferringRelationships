"""Defines the shifting attention referring relationship model.
"""

from config import parse_args
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, UpSampling2D, Input, Activation
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.core import Lambda, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Multiply, Dot, Add
from keras.models import Model

import numpy as np



class ReferringRelationshipsModel():
    """Given a relationship, this model locatlizes them.
    """

    def __init__(self, args):
        """Constructor for ReferringRelationshipModel.

        Args:
            args: the arguments specified by `config.py`.
        """
        self.input_dim = args.input_dim
        self.feat_map_dim = args.feat_map_dim
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim
        self.num_predicates = args.num_predicates
        self.num_objects = args.num_objects
        self.dropout = args.dropout
        self.use_subject = args.use_subject
        self.use_predicate = args.use_predicate
        self.use_object = args.use_object
        self.conv_predicate_kernel = args.conv_predicate_kernel
        self.use_conv = args.use_conv_ssn
        self.nb_conv_move_map = args.nb_conv_move_map
        self.feat_map_layer = args.feat_map_layer

    def _build_model(self):
        """Initializes the SSN model.
        This model uses moving heatmaps with a dense or conv layer
        Returns:
            The Keras model.
        """
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_pred = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        im_features = self.build_image_model(input_im)
        im_features = Dropout(self.dropout)(im_features)
        im_features = Conv2D(self.hidden_dim, 1, padding='same', activation="relu")(im_features)
        im_features = Dropout(self.dropout)(im_features)
        subj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        if self.use_conv:
            predicate_embedding = self.build_embedding_layer(self.num_predicates, self.conv_predicate_kernel**2)
        else:
            predicate_embedding = self.build_embedding_layer(self.num_predicates, self.feat_map_dim**4)
        embedded_subject = subj_embedding(input_subj)
        embedded_subject = Dropout(self.dropout)(embedded_subject)
        embedded_predicate = predicate_embedding(input_pred)
        embedded_predicate = Dropout(self.dropout)(embedded_predicate)
        embedded_object = obj_embedding(input_obj)
        embedded_object = Dropout(self.dropout)(embedded_object)
        subject_att = self.build_attention_layer(im_features, embedded_subject, "before-pred")
        subject_regions = self.build_upsampling_layer(subject_att, "subject")
        if self.use_conv:
            predicate_att = self.build_map_transform_layer_conv(subject_att, embedded_predicate, "after-pred")
        else:
            predicate_att = self.build_map_transform_layer_dense(subject_att, embedded_predicate, "after-pred")
        predicate_att = Activation("tanh")(predicate_att)
        new_im_feature_map = Multiply()([im_features, predicate_att])
        object_att = self.build_attention_layer(new_im_feature_map, embedded_object)
        object_regions = self.build_upsampling_layer(object_att, "object")
        model = Model(inputs=[input_im, input_subj, input_pred, input_obj], outputs=[subject_regions, object_regions])
        return model

    def build_model(self): 
        """Initializes the SSN model.
        This model uses moving heatmaps with a dense layer
        Returns:
            The Keras model.
        """
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_pred = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        im_features = self.build_image_model(input_im)
        im_features = Dropout(self.dropout)(im_features)
        im_features = Conv2D(self.hidden_dim, 1, padding='same', activation="relu")(im_features)
        im_features = Dropout(self.dropout)(im_features)
        subj_obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        if self.use_conv:
            predicate_embedding = self.build_embedding_layer(self.num_predicates, self.conv_predicate_kernel**2)
            inverse_predicate_embedding = self.build_embedding_layer(self.num_predicates, self.conv_predicate_kernel**2)
        else:
            predicate_embedding = self.build_embedding_layer(self.num_predicates, self.feat_map_dim**4)
            inverse_predicate_embedding = self.build_embedding_layer(self.num_predicates, self.feat_map_dim**4)
        embedded_subject = subj_obj_embedding(input_subj)
        embedded_subject = Dropout(self.dropout)(embedded_subject)
        embedded_predicate = predicate_embedding(input_pred)
        embedded_predicate = Dropout(self.dropout)(embedded_predicate)
        embedded_inverse_predicate = inverse_predicate_embedding(input_pred)
        embedded_inverse_predicate = Dropout(self.dropout)(embedded_inverse_predicate)
        embedded_object = subj_obj_embedding(input_obj)
        embedded_object = Dropout(self.dropout)(embedded_object)
        subject_att = self.build_attention_layer(im_features, embedded_subject, "before-pred-subj")
        object_att = self.build_attention_layer(im_features, embedded_object, "before-pred-obj")
        if self.use_conv:
            subj_predicate_att = self.build_map_transform_layer_conv(subject_att, embedded_predicate, "after-pred-subj")
            obj_predicate_att = self.build_map_transform_layer_conv(object_att, embedded_inverse_predicate, "after-pred-obj")
        else:
            subj_predicate_att = self.build_map_transform_layer_dense(subject_att, embedded_predicate, "after-pred-subj")
            obj_predicate_att = self.build_map_transform_layer_dense(object_att, embedded_inverse_predicate, "after-pred-obj")
        subj_predicate_att = Activation("tanh")(subj_predicate_att)
        obj_predicate_att = Activation("tanh")(obj_predicate_att)
        attended_im_subj = Multiply()([im_features, subj_predicate_att])
        attended_im_obj = Multiply()([im_features, obj_predicate_att])
        object_att = self.build_attention_layer(attended_im_subj, embedded_object)
        subject_att = self.build_attention_layer(attended_im_obj, embedded_subject)
        object_regions = self.build_upsampling_layer(object_att, "object")
        subject_regions = self.build_upsampling_layer(subject_att, "subject")
        model = Model(inputs=[input_im, input_subj, input_pred, input_obj], outputs=[subject_regions, object_regions])
        return model

    def build_image_model(self, input_im):
        """Grab the image features.

        Args:
            input_im: The input image to the model.

        Returns:
            The image feature map.
        """
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(self.input_dim, self.input_dim, 3))
        for layer in base_model.layers:
            layer.trainable = False
        output = base_model.get_layer(self.feat_map_layer).output
        image_branch = Model(inputs=base_model.input, outputs=output)
        im_features = image_branch(input_im)
        return im_features

    def build_map_transform_layer_dense(self, att_weights, pred_features, name):
        att_weights_flat = Reshape((1, self.feat_map_dim**2))(att_weights) # N x H
        pred_matrix = Reshape((self.feat_map_dim**2, self.feat_map_dim**2))(pred_features) # H x H
        att_transformed = Multiply()([att_weights_flat, pred_matrix])
        att_transformed = Lambda(lambda x: K.sum(x, axis=2))(att_transformed)
        att_transformed = Reshape((self.feat_map_dim, self.feat_map_dim, 1), name=name)(att_transformed)
        return att_transformed

    def build_map_transform_layer_conv(self, att_map, kernel, name):
        # This function assumes that all predicates are the same within a batch
        kernel = Lambda(lambda x: K.mean(x, axis=0))(kernel)
        kernel = Lambda(lambda x: K.reshape(x, (self.conv_predicate_kernel, self.conv_predicate_kernel, 1, 1)))(kernel)
        for i in range(self.nb_conv_move_map-1):
            att_map = Lambda(lambda x: K.conv2d(x[0], x[1], padding='same'))([att_map, kernel])
        att_transformed = Lambda(lambda x: K.conv2d(x[0], x[1], padding='same'), name=name)([att_map, kernel])
        return att_transformed

    def build_embedding_layer(self, num_categories, emb_dim):
        return Embedding(num_categories, emb_dim, input_length=1)

    def build_attention_layer(self, feature_map, query, name=None):
        query = Reshape((1, 1, self.hidden_dim))(query)
        attention_weights = Multiply()([feature_map, query])
        if not name:
            attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(attention_weights)
        else:
             attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True), name=name)(attention_weights)
        return attention_weights

    def build_frac_strided_transposed_conv_layer(self, conv_layer):
        res = UpSampling2D(size=(2, 2))(conv_layer)
        res = Conv2DTranspose(1, 3, padding='same')(res)
        return res

    def build_upsampling_layer(self, feature_map, name):
        upsampling_factor = self.input_dim / self.feat_map_dim
        k = int(np.log(upsampling_factor) / np.log(2))
        res = feature_map
        for i in range(k):
            res = self.build_frac_strided_transposed_conv_layer(res)
        predictions = Activation('sigmoid')(res)
        predictions = Flatten(name=name)(predictions)
        return predictions

    def build_model_1(self):
        """Initializes the SSN model.
        This model uses refined query for attention.
        No predicate
        Similar to stacked attention: uses the same image feature map for each attention layer but different query vector
        Returns:
            The Keras model.
        """
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_pred = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        im_features = self.build_image_model(input_im)
        im_features = Conv2D(self.hidden_dim, 1, padding='same', activation="relu")(im_features)
        subj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        embedded_subject = subj_embedding(input_subj)
        embedded_object = obj_embedding(input_obj)
        subject_att = self.build_attention_layer(im_features, embedded_subject)
        subject_regions = self.build_upsampling_layer(subject_att)
        subject_regions_flat = Reshape((self.input_dim*self.input_dim,), name="subject")(subject_regions)
        refined_query = self.build_refined_query(im_features, subject_att, embedded_object, "after-pred")
        object_att = self.build_attention_layer(im_features, refined_query)
        object_regions = self.build_upsampling_layer(object_att)
        object_regions_flat = Reshape((self.input_dim*self.input_dim,), name="object")(object_regions)
        model = Model(inputs=[input_im, input_subj, input_pred, input_obj], outputs=[subject_regions_flat, object_regions_flat])
        return model

    def build_refined_query(self, im_features, subject_att, embedded_object):
        x1 = Multiply()([im_features, subject_att])
        x2 = Reshape((self.feat_map_dim*self.feat_map_dim, self.hidden_dim))(x1)
        object_query = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(x2)
        refined_query =  Add()([object_query, embedded_object])
        return refined_query

    def build_model_2(self):
        """Initializes the SSN model.
        This model precits two heatmaps for obj and subj.
        No stacked attention.
        Only for debugging.
        Returns:
            The Keras model.
        """
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_pred = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        im_features = self.build_image_model(input_im)
        im_features = Conv2D(self.hidden_dim, 1, padding='same', activation="relu")(im_features)
        subj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        embedded_subject = subj_embedding(input_subj)
        embedded_object = obj_embedding(input_obj)
        subject_att = self.build_attention_layer(im_features, embedded_subject)
        object_att = self.build_attention_layer(im_features, embedded_object)
        subject_regions = self.build_upsampling_layer(subject_att)
        object_regions = self.build_upsampling_layer(object_att)
        subject_regions_flat = Flatten(name="subject")(subject_regions)
        object_regions_flat = Flatten(name="object")(object_regions)
        model = Model(inputs=[input_im, input_subj, input_pred, input_obj], outputs=[subject_regions_flat, object_regions_flat])
        return model


if __name__ == "__main__":
    args = parse_args()
    rel = ReferringRelationshipsModel(args)
    model = rel.build_model()
    model.summary()
