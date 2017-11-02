"""Defines the shifting attention referring relationship model.
"""

from config import parse_args
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, UpSampling2D, Input, Activation
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.core import Lambda, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Multiply, Dot, Add, Concatenate
from keras.models import Model
from keras.regularizers import l2

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
        self.num_predicates = args.num_predicates
        self.num_objects = args.num_objects
        self.dropout = args.dropout
        self.use_subject = args.use_subject
        self.use_predicate = args.use_predicate
        self.use_object = args.use_object
        self.nb_conv_att_map = args.nb_conv_att_map
        self.nb_conv_im_map = args.nb_conv_im_map
        self.feat_map_layer = args.feat_map_layer
        self.conv_im_kernel = args.conv_im_kernel
        self.conv_predicate_kernel = args.conv_predicate_kernel
        self.conv_predicate_channels = args.conv_predicate_channels
        self.model = args.model
        self.reg = args.reg
        self.nb_dense_emb = args.nb_dense_emb
        self.use_internal_loss = args.use_internal_loss
        self.att_activation = args.att_activation
        self.internal_loss_weight = args.internal_loss_weight
        self.att_mechanism = args.att_mechanism

    def build_model(self):
        if self.model=="ssn":
            return self.build_ssn_model()
        elif self.model=="sym_ssn":
            return self.build_sym_ssn_model()
        elif self.model=="baseline":
            return self.build_baseline_model()
        elif self.model=="baseline_no_predicate":
            return self.build_baseline_model_no_predicate()
        else:
            raise ValueError("model argument not recognized. Model options: ssn, sym_ssn, baseline, baseline_no_predicte")

    def build_ssn_model(self):
        """Initializes the stacked attention model.
        This model uses moving heatmaps with conv layers
        Returns:
            The Keras model.
        """
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_pred = Input(shape=(self.num_predicates,))
        input_obj = Input(shape=(1,))
        im_features = self.build_image_model(input_im)
        subj_obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        embedded_subject = subj_obj_embedding(input_subj)
        embedded_subject = Dropout(self.dropout)(embedded_subject)
        embedded_object = subj_obj_embedding(input_obj)
        embedded_object = Dropout(self.dropout)(embedded_object)
        if self.att_mechanism == "mul":
            subject_att = self.build_attention_layer_mul(im_features, embedded_subject)
            subject_att_pooled = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(subject_att)
            subject_regions = self.build_upsampling_layer(subject_att_pooled, "subject")
        else:
            subject_att = self.build_attention_layer_dot(im_features, embedded_subject, "before-pred")
            subject_regions = self.build_upsampling_layer(subject_att, "subject")
        predicate_att = self.build_conv_map_transform(subject_att, input_pred, "after-pred")
        new_im_feature_map = Multiply()([im_features, predicate_att])
        object_att = self.build_attention_layer_dot(new_im_feature_map, embedded_object)
        object_regions = self.build_upsampling_layer(object_att, "object")
        model = Model(inputs=[input_im, input_subj, input_pred, input_obj], outputs=[subject_regions, object_regions])
        return model


    def build_sym_ssn_model(self):
        """Initializes the symmetric stacked attention model
        This model uses moving heatmaps with conv layers
        Returns:
            The Keras model.
        """
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_pred = Input(shape=(self.num_predicates,))
        input_obj = Input(shape=(1,))
        im_features = self.build_image_model(input_im)
        subj_obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        embedded_subject = subj_obj_embedding(input_subj)
        embedded_subject = Dropout(self.dropout)(embedded_subject)
        embedded_object = subj_obj_embedding(input_obj)
        embedded_object = Dropout(self.dropout)(embedded_object)
        if self.att_mechanism == "mul":
            subject_att = self.build_attention_layer_mul(im_features, embedded_subject)
            object_att = self.build_attention_layer_mul(im_features, embedded_object)
            if self.use_internal_loss:
                subject_att_pooled = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(subject_att)
                subject_regions_int = self.build_upsampling_layer(subject_att_pooled, "subject-int")
                object_att_pooled = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(object_att)
                object_regions_int = self.build_upsampling_layer(object_att_pooled, "object-int")
        else:
            subject_att = self.build_attention_layer_dot(im_features, embedded_subject)
            object_att = self.build_attention_layer_dot(im_features, embedded_object)
            if self.use_internal_loss:
                subject_regions_int = self.build_upsampling_layer(subject_att, "subject-int")
                object_regions_int = self.build_upsampling_layer(object_att, "object-int")
        subj_predicate_att = self.build_conv_map_transform(subject_att, input_pred, "after-pred-subj")
        obj_predicate_att = self.build_conv_map_transform(object_att, input_pred, "after-pred-obj", sym=1)
        attended_im_subj = Multiply()([im_features, subj_predicate_att])
        attended_im_obj = Multiply()([im_features, obj_predicate_att])
        object_att = self.build_attention_layer_dot(attended_im_subj, embedded_object)
        subject_att = self.build_attention_layer_dot(attended_im_obj, embedded_subject)
        if self.use_internal_loss:
            subject_regions = self.build_upsampling_layer(subject_att, "subject-out")
            object_regions = self.build_upsampling_layer(object_att, "object-out")
            subject_regions = Lambda(lambda x: (1. - self.internal_loss_weight)*x[0] + self.internal_loss_weight*x[1], name="subject")([subject_regions, subject_regions_int])
            object_regions = Lambda(lambda x: (1. - self.internal_loss_weight)*x[0] + self.internal_loss_weight*x[1], name="object")([object_regions, object_regions_int])
        else:
           object_regions = self.build_upsampling_layer(object_att, "object")
           subject_regions = self.build_upsampling_layer(subject_att, "subject")
        outputs = [subject_regions, object_regions]
        model = Model(inputs=[input_im, input_subj, input_pred, input_obj], outputs=outputs)
        return model

    def build_baseline_model(self):
        """Initializes the baseline model that uses predicates.
        Returns:
            The Keras model.
        """

        # Setup the inputs.
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        relationship_inputs = []
        num_classes = []
        if self.use_subject:
            input_sub = Input(shape=(1,))
            relationship_inputs.append(input_sub)
            num_classes.append(self.num_objects)
        if self.use_predicate:
            input_pred = Input(shape=(1,))
            relationship_inputs.append(input_pred)
            num_classes.append(self.num_predicates)
        if self.use_object:
            input_obj = Input(shape=(1,))
            relationship_inputs.append(input_obj)
            num_classes.append(self.num_objects)

        # Map the inputs to the outputs.
        im_features = self.build_image_model(input_im)
        rel_features = self.build_relationship_model(relationship_inputs, num_classes)
        rel_features = Dropout(self.dropout)(rel_features)
        subjects_features = Dense(self.hidden_dim)(rel_features)
        objects_features = Dense(self.hidden_dim)(rel_features)
        subjects_features = Dropout(self.dropout)(subjects_features)
        objects_features = Dropout(self.dropout)(objects_features)
        subject_att = self.build_attention_layer_dot(im_features, subjects_features)
        object_att = self.build_attention_layer_dot(im_features, objects_features)
        subject_regions = self.build_upsampling_layer(subject_att, "subject")
        object_regions = self.build_upsampling_layer(object_att, "object")
        model_inputs = [input_im] + relationship_inputs
        model = Model(inputs=model_inputs, outputs=[subject_regions, object_regions])
        return model

    def build_baseline_model_no_predicate(self):
        """Initializes the SSN model.
        This model precits two heatmaps for obj and subj.
        No stacked attention.
        This baseline does not use the predicate.
        Returns:
            The Keras model.
        """
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        im_features = self.build_image_model(input_im)
        subj_obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        embedded_subject = subj_obj_embedding(input_subj)
        embedded_subject = Dropout(self.dropout)(embedded_subject)
        embedded_object = subj_obj_embedding(input_obj)
        embedded_object = Dropout(self.dropout)(embedded_object)
        subject_att = self.build_attention_layer_dot(im_features, embedded_subject)
        object_att = self.build_attention_layer_dot(im_features, embedded_object)
        subject_regions = self.build_upsampling_layer(subject_att, "subject")
        object_regions = self.build_upsampling_layer(object_att, "object")
        model = Model(inputs=[input_im, input_subj, input_obj], outputs=[subject_regions, object_regions])
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
        im_features = Dropout(self.dropout)(im_features)
        for i in range(self.nb_conv_im_map-1):
            im_features = Conv2D(self.hidden_dim, self.conv_im_kernel, strides=(1, 1), padding='same', activation='relu')(im_features)
            im_features = Dropout(self.dropout)(im_features)
        im_features = Conv2D(self.hidden_dim, self.conv_im_kernel, strides=(1, 1), padding='same')(im_features)
        im_features = Dropout(self.dropout)(im_features)
        return im_features

    def build_embedding_layer(self, num_categories, emb_dim):
        return Embedding(num_categories, emb_dim, input_length=1)

    def build_attention_layer_mul(self, feature_map, query):
        query = Reshape((1, 1, self.hidden_dim))(query)
        attention_weights = Multiply()([feature_map, query])
        return attention_weights

    def build_attention_layer_dot(self, feature_map, query, name=None):
        query = Reshape((1, 1, self.hidden_dim))(query)
        attention_weights = Multiply()([feature_map, query])
        if not name:
            attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(attention_weights)
        else:
             attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True), name=name)(attention_weights)
        return attention_weights

    def build_conv_predicate_module(self, att_map, predicate_id, sym):
        for i in range(self.nb_conv_att_map):
            att_map = Conv2D(self.conv_predicate_channels, self.conv_predicate_kernel, strides=(1, 1), padding='same', use_bias=False, name='conv{}-predicate{}-{}'.format(i, predicate_id, sym))(att_map)
        shifted_att = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(att_map)
        return shifted_att

    def build_conv_map_transform(self, att, input_pred, name, sym=0):
        predicate_masks = Reshape((1, 1, self.num_predicates))(input_pred)
        conv_modules = []
        for i in range(self.num_predicates):
            conv_modules += [self.build_conv_predicate_module(att, i, sym)]
        merged_conv = Concatenate(axis=3)(conv_modules)
        att = Multiply()([predicate_masks, merged_conv])
        att = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(att)
        if self.att_activation == "tanh":
            predicate_att = Activation("tanh", name=name)(att)
        elif self.att_activation == "tanh+relu":
            predicate_att = Activation("tanh")(att)
            predicate_att = Activation("relu", name=name)(predicate_att)
        elif self.att_activation == "norm":
            att = Reshape((self.feat_map_dim*self.feat_map_dim,))(att)
            att = Lambda(lambda x: x-K.min(x, axis=1, keepdims=True))(att)
            att = Lambda(lambda x: x/(K.epsilon() + K.max(K.abs(x), axis=1, keepdims=True)))(att)
            predicate_att = Reshape((self.feat_map_dim, self.feat_map_dim, 1), name=name)(att)
        elif self.att_activation == "norm+relu":
            att = Reshape((self.feat_map_dim*self.feat_map_dim,))(att)
            att = Lambda(lambda x: x-K.mean(x, axis=1, keepdims=True))(att)
            att = Lambda(lambda x: x/(K.epsilon() + K.max(K.abs(x), axis=1, keepdims=True)))(att)
            predicate_att = Reshape((self.feat_map_dim, self.feat_map_dim, 1))(att)
            predicate_att =  Activation("relu", name=name)(predicate_att)
        elif self.att_activation == "binary":
            att = Reshape((self.feat_map_dim*self.feat_map_dim,))(att)
            att = Lambda(lambda x: x-K.mean(x, axis=1, keepdims=True))(att)
            predicate_att =  Lambda(lambda x: K.cast(K.greater(x, 0), K.floatx()))(att)
        return predicate_att

    def build_frac_strided_transposed_conv_layer(self, conv_layer):
        res = UpSampling2D(size=(2, 2))(conv_layer)
        res = Conv2DTranspose(1, 3, padding='same', use_bias=False)(res)
        return res

    def build_upsampling_layer(self, feature_map, name):
        upsampling_factor = self.input_dim / self.feat_map_dim
        k = int(np.log(upsampling_factor) / np.log(2))
        res = Activation("relu")(feature_map)
        for i in range(k):
            res = self.build_frac_strided_transposed_conv_layer(res)
        res = Reshape((self.input_dim*self.input_dim,))(res)
        predictions = Activation("tanh", name=name)(res)
        return predictions

    def build_relationship_model(self, relationship_inputs, num_classes):
        """Converts the input relationship into a feature space.
        Args:
            relationship_inputs: A list of inputs to the model. Can contains
              input_sub, input_pred or input_obj depending on the args.
            num_classes: A list containing how many categories each input can
              take in the relatinoship_inputs list. Used to initialize the
              embedding layer.
        Returns:
            The feature representation for the relationship.
        """
        embeddings = []
        for rel_input, num_categories in zip(relationship_inputs, num_classes):
            embedding_layer = Embedding(num_categories,
                                        int(self.hidden_dim/len(relationship_inputs)),
                                        input_length=1)
            embeddings.append(embedding_layer(rel_input))

        # Concatenate the inputs if there are more than 1.
        if len(embeddings) > 1:
            concatenated_inputs = Concatenate(axis=2)(embeddings)
        else:
            concatenated_inputs = embeddings[0]
        concatenated_inputs = Dropout(self.dropout)(concatenated_inputs)
        rel_features = Dense(self.hidden_dim, activation='relu')(concatenated_inputs)
        return rel_features

if __name__ == "__main__":
    args = parse_args()
    rel = ReferringRelationshipsModel(args)
    model = rel.build_model()
    model.summary()
