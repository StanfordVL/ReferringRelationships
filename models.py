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
        self.model = args.model
        self.reg = args.reg
        self.nb_dense_emb = args.nb_dense_emb
        self.use_internal_loss = args.use_internal_loss
        self.att_activation = args.att_activation

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
        subject_att = self.build_attention_layer(im_features, embedded_subject, "before-pred")
        subject_regions = self.build_upsampling_layer(subject_att, "subject")
        predicate_att = self.build_conv_map_transform(subject_att, input_pred, "after-pred")
        new_im_feature_map = Multiply()([im_features, predicate_att])
        object_att = self.build_attention_layer(new_im_feature_map, embedded_object)
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
        subject_att = self.build_attention_layer(im_features, embedded_subject, "before-pred-subj")
        object_att = self.build_attention_layer(im_features, embedded_object, "before-pred-obj")
        if self.use_internal_loss:
            subject_regions_int = self.build_upsampling_layer(subject_att, "subject-int")
            object_regions_int = self.build_upsampling_layer(object_att, "object-int")    
        subj_predicate_att = self.build_conv_map_transform(subject_att, input_pred, "after-pred-subj")
        obj_predicate_att = self.build_conv_map_transform(object_att, input_pred, "after-pred-obj")
        attended_im_subj = Multiply()([im_features, subj_predicate_att])
        attended_im_obj = Multiply()([im_features, obj_predicate_att])
        object_att = self.build_attention_layer(attended_im_subj, embedded_object)
        subject_att = self.build_attention_layer(attended_im_obj, embedded_subject)
        object_regions = self.build_upsampling_layer(object_att, "object")
        subject_regions = self.build_upsampling_layer(subject_att, "subject")
        if self.use_internal_loss:
            outputs = [subject_regions_int, object_regions_int, subject_regions, object_regions]
        else:
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
        subject_att = self.build_attention_layer(im_features, subjects_features)
        object_att = self.build_attention_layer(im_features, objects_features)
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
        subject_att = self.build_attention_layer(im_features, embedded_subject)
        object_att = self.build_attention_layer(im_features, embedded_object)
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

    def build_attention_layer(self, feature_map, query, name=None):
        query = Reshape((1, 1, self.hidden_dim))(query)
        attention_weights = Multiply()([feature_map, query])
        if not name:
            attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(attention_weights)
        else:
             attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True), name=name)(attention_weights)
        return attention_weights

    def build_conv_map_transform(self, att, input_pred, name):
        predicate_masks = Reshape((1, 1, self.num_predicates))(input_pred)
        for i in range(self.nb_conv_att_map):
            att = Conv2D(self.num_predicates, self.conv_predicate_kernel, strides=(1, 1), padding='same')(att)
            att = Multiply()([predicate_masks, att])
            att = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(att)
        if self.att_activation == "tanh":
            predicate_att = Activation("tanh", name=name)(att)
        elif self.att_activation == "tanh+relu":
            predicate_att = Activation("tanh")(att)
            predicate_att = Activation("relu", name=name)(predicate_att)
        elif self.att_activation == "norm":
            att = Reshape((self.feat_map_dim*self.feat_map_dim,))(att)
            att = Lambda(lambda x: K.l2_normalize(x, axis=1))(att)
            predicate_att = Reshape((self.feat_map_dim, self.feat_map_dim, 1))(att)
        else:
            predicate_att =  Lambda(lambda x: K.cast(K.greater(x, 0), K.floatx()))(att)
        return predicate_att

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
        subj_obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        embedded_subject = subj_obj_embedding(input_subj)
        embedded_object = subj_obj_embedding(input_obj)
        subject_att = self.build_attention_layer(im_features, embedded_subject)
        subject_regions = self.build_upsampling_layer(subject_att, "subject")
        refined_query = self.build_refined_query(im_features, subject_att, embedded_object, "after-pred")
        object_att = self.build_attention_layer(im_features, refined_query)
        object_regions = self.build_upsampling_layer(object_att, "object")
        model = Model(inputs=[input_im, input_subj, input_pred, input_obj], outputs=[subject_regions_flat, object_regions_flat])
        return model

    def build_refined_query(self, im_features, subject_att, embedded_object):
        x1 = Multiply()([im_features, subject_att])
        x2 = Reshape((self.feat_map_dim*self.feat_map_dim, self.hidden_dim))(x1)
        object_query = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(x2)
        refined_query =  Add()([object_query, embedded_object])
        return refined_query

if __name__ == "__main__":
    args = parse_args()
    rel = ReferringRelationshipsModel(args)
    model = rel.build_model()
    model.summary()
