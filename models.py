"""Defines the shifting attention referring relationship model.
"""

from config import parse_args
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Input, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.layers.merge import Multiply
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
        self.num_objects = args.num_objects
        self.num_predicates = args.num_predicates
        self.dropout = args.dropout
        self.use_subject = args.use_subject
        self.use_predicate = args.use_predicate
        self.use_object = args.use_object
        self.nb_conv_att_map = args.nb_conv_att_map
        self.nb_conv_im_map = args.nb_conv_im_map
        self.cnn = args.cnn
        self.feat_map_layer = args.feat_map_layer
        self.conv_im_kernel = args.conv_im_kernel
        self.conv_predicate_kernel = args.conv_predicate_kernel
        self.conv_predicate_channels = args.conv_predicate_channels
        self.model = args.model
        self.use_internal_loss = args.use_internal_loss
        self.internal_loss_weight = args.internal_loss_weight
        self.iterations = args.iterations
        self.attention_conv_kernel = args.attention_conv_kernel
        self.refinement_conv_kernel = args.refinement_conv_kernel
        self.output_dim = args.output_dim
        self.embedding_dim = args.embedding_dim
        self.finetune_cnn = args.finetune_cnn

        # Discovery. Create a general object attention embedding option.
        if args.discovery:
            self.num_objects += 1

    def build_model(self):
        """Chooses which model based on the arguments.
        """
        if self.model == "co-occurrence":
            if self.use_predicate:
                raise ValueError('co-occurrence model must be run with '
                                 '--use-predicate False.')
            return self.build_vrd()
        elif self.model == "vrd":
            if not self.use_predicate:
                raise ValueError('vrd model must be run with '
                                 '--use-predicate True.')
            return self.build_vrd()
        elif self.model == "ssas":
            return self.build_ssas()
        else:
            raise ValueError('Model argument not recognized. Model options: '
                             'ssas, co-occurrence, vrd.')

    def build_ssas(self):
        """Focusing on the subject and object iteratively.

        Returns:
            The Keras model.
        """
        # Create Inputs.
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        if self.use_predicate:
            input_pred = Input(shape=(self.num_predicates,))
            inputs=[input_im, input_subj, input_pred, input_obj]
        else:
            inputs=[input_im, input_subj, input_obj]

        # Generate the embedding and image feature layers.
        im_features = self.build_image_model(input_im)
        subj_obj_embedding = self.build_embedding_layer(
            self.num_objects, self.embedding_dim)
        embedded_subject = subj_obj_embedding(input_subj)
        embedded_subject = Dense(self.hidden_dim, activation="relu")(
            embedded_subject)
        embedded_subject = Dropout(self.dropout)(embedded_subject)
        embedded_object = subj_obj_embedding(input_obj)
        embedded_object = Dense(self.hidden_dim, activation="relu")(
            embedded_object)
        embedded_object = Dropout(self.dropout)(embedded_object)

        # Initial attention over the subject and object.
        subject_att = self.attend(im_features, embedded_subject, name='subject-att-0')
        object_att = self.attend(im_features, embedded_object, name='object-att-0')

        # Create the predicate conv layers.
        if self.use_predicate:
            predicate_masks = Reshape((1, 1, self.num_predicates))(input_pred)
            predicate_modules = self.build_conv_modules(
                basename='conv{}-predicate{}')
            inverse_predicate_modules = self.build_conv_modules(
                basename='conv{}-inv-predicate{}')

        # Save the initial predictions when using the internal loss.
        if self.use_internal_loss:
            subject_outputs = [subject_att]
            object_outputs = [object_att]

        # Iterate!
        for iteration in range(self.iterations):
            predicate_att = self.shift_conv_attention(subject_att, predicate_modules, predicate_masks)
            predicate_att = Lambda(lambda x: x, name='shift-{}'.format(iteration+1))(predicate_att)
            new_image_features = Multiply()([im_features, predicate_att])
            new_object_att = self.attend(new_image_features, embedded_object, name='object-att-{}'.format(iteration+1))

            inv_predicate_att = self.shift_conv_attention(object_att, inverse_predicate_modules, predicate_masks)
            inv_predicate_att = Lambda(lambda x: x, name='inv-shift-{}'.format(iteration+1))(inv_predicate_att)
            new_image_features = Multiply()([im_features, inv_predicate_att])
            new_subject_att = self.attend(new_image_features, embedded_subject, name='subject-att-{}'.format(iteration+1))

            if self.use_internal_loss:
                object_outputs.append(new_object_att)
                subject_outputs.append(new_subject_att)
            object_att = new_object_att
            subject_att = new_subject_att

        # Collect the subject and objects regions.
        if self.use_internal_loss and self.iterations > 0:
            internal_weights = np.array([self.internal_loss_weight**i for i in range(len(subject_outputs))])
            internal_weights = internal_weights / internal_weights.sum()

            # Multiply with the internal losses.
            subject_outputs = Lambda(lambda x: [internal_weights[i]*x[i] for i in range(len(x))])(subject_outputs)
            object_outputs = Lambda(lambda x: [internal_weights[i]*x[i] for i in range(len(x))])(object_outputs)

            # Concatenate all the internal outputs.
            subject_att = Concatenate(axis=3)(subject_outputs)
            object_att = Concatenate(axis=3)(object_outputs)

            # Sum across the internal values.
            subject_att = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(subject_att)
            object_att = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(object_att)

        subject_att = Activation("tanh")(subject_att)
        object_att = Activation("tanh")(object_att)
        subject_regions = Reshape((self.output_dim * self.output_dim,), name="subject")(subject_att)
        object_regions = Reshape((self.output_dim * self.output_dim,), name="object")(object_att)

        # Create and output the model.
        model = Model(inputs=inputs, outputs=[subject_regions, object_regions])
        return model

    def build_conv_modules(self, basename):
        """Creates the convolution modules used to shift attention.

        Args:
            basename: String representing the name of the conv.

        Returns:
            A list of length `self.num_predicates` convolution modules.
        """
        predicate_modules = []
        for k in range(self.num_predicates):
            predicate_module_group = []
            for i in range(self.nb_conv_att_map-1):
                predicate_conv = Conv2D(
                    self.conv_predicate_channels, self.conv_predicate_kernel,
                    strides=(1, 1), padding='same', use_bias=False,
                    activation='relu',
                    name=basename.format(i, k))
                predicate_module_group.append(predicate_conv)
            # last conv with only one channel
            predicate_conv = Conv2D(
                1, self.conv_predicate_kernel,
                strides=(1, 1), padding='same', use_bias=False,
                activation='relu',
                name=basename.format(self.nb_conv_att_map-1, k))
            predicate_module_group.append(predicate_conv)
            predicate_modules.append(predicate_module_group)
        return predicate_modules

    def shift_conv_attention(self, att, merged_modules, predicate_masks):
        """Takes an intial attention map and shifts it using the predicate convs.

        Args:
            att: An initial attention by the object or the subject.
            merged_modules: A list containing `self.num_predicate` elements
              where each element is a list of `self.nb_conv_att_map` convs.
            predicate_masks: A helpful tensor indicating which predicate
              is involved for relationship element in the batch.

        Returns:
            The shifted attention.
        """
        conv_outputs = []
        for group in merged_modules:
            att_map = att
            for conv_module in group:
                att_map = conv_module(att_map)
            conv_outputs.append(att_map)
        merged_output = Concatenate(axis=3)(conv_outputs)
        predicate_att = Multiply()([predicate_masks, merged_output])
        predicate_att = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(predicate_att)
        predicate_att = Activation("tanh")(predicate_att)
        return predicate_att

    def build_vrd(self):
        """Initializes the baseline model that uses predicates.

        Returns:
            The Keras model.
        """
        # Setup the inputs.
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        relationship_inputs = []
        num_classes = []
        input_sub = Input(shape=(1,))
        relationship_inputs.append(input_sub)
        num_classes.append(self.num_objects)
        if self.use_predicate:
            input_pred = Input(shape=(1,))
            relationship_inputs.append(input_pred)
            num_classes.append(self.num_predicates)
        input_obj = Input(shape=(1,))
        relationship_inputs.append(input_obj)
        num_classes.append(self.num_objects)

        # Grab the image features.
        im_features = self.build_image_model(input_im)

        # Embed the relationship.
        rel_features = self.build_relationship_model(relationship_inputs, num_classes)
        rel_features = Dropout(self.dropout)(rel_features)
        subjects_features = Dense(self.hidden_dim, activation="relu")(rel_features)
        objects_features = Dense(self.hidden_dim, activation="relu")(rel_features)
        subjects_features = Dropout(self.dropout)(subjects_features)
        objects_features = Dropout(self.dropout)(objects_features)

        # Attend over the image regions.
        subject_att = self.attend(im_features, subjects_features)
        object_att = self.attend(im_features, objects_features)
        subject_att = Activation("tanh")(subject_att)
        object_att = Activation("tanh")(object_att)

        # Output the predictions.
        subject_regions = Reshape((self.output_dim * self.output_dim,), name="subject")(subject_att)
        object_regions = Reshape((self.output_dim * self.output_dim,), name="object")(object_att)
        model_inputs = [input_im] + relationship_inputs
        model = Model(inputs=model_inputs, outputs=[subject_regions, object_regions])
        return model

    def build_image_model(self, input_im):
        """Grab the image features.

        Finetunes the image feature layers if self.finetune_cnn = True.

        Args:
            input_im: The input image to the model.

        Returns:
            The image feature map.
        """
        if self.cnn == "resnet":
            base_model = ResNet50(
                weights='imagenet', include_top=False,
                input_shape=(self.input_dim, self.input_dim, 3))
        elif self.cnn == "vgg":
            base_model = VGG19(
                weights='imagenet', include_top=False,
                input_shape=(self.input_dim, self.input_dim, 3))
        else:
            raise ValueError('--cnn must be [resnet, vgg] but got {}'.format(
                self.cnn))
        if self.finetune_cnn:
            for layer in base_model.layers:
                layer.trainable = True
                layer.training = True
        else:
            for layer in base_model.layers:
                layer.trainable = False
                layer.training = False
        output = base_model.get_layer(self.feat_map_layer).output
        image_branch = Model(inputs=base_model.input, outputs=output)
        im_features = image_branch(input_im)
        im_features = Dropout(self.dropout)(im_features)
        for i in range(self.nb_conv_im_map):
            im_features = Conv2D(self.hidden_dim, self.conv_im_kernel,
                                 strides=(1, 1), padding='same',
                                 activation='relu')(im_features)
            im_features = Dropout(self.dropout)(im_features)
        return im_features

    def build_embedding_layer(self, num_categories, emb_dim):
        """Creates an embedding layer.

        Args:
            num_categories: The number of categories we want embeddings for.
            emb_dim: The dimensions in the embedding.

        Returns:
            An embedding layer.
        """
        return Embedding(num_categories, emb_dim, input_length=1)

    def attend(self, feature_map, query, name=None):
        """Uses the embedded query to attend over the image features.

        Args:
            feature_map: The image features to attend over.
            query: The embedding of the category used as the attention query.
            name: The name of the layer.

        Returns:
            The attention weights over the feature map.
        """
        query = Reshape((1, 1, self.hidden_dim,))(query)
        attention_weights = Multiply()([feature_map, query])
        attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(attention_weights)
        attention_weights = Activation("relu", name=name)(attention_weights)
        return attention_weights

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
                                        self.embedding_dim,
                                        input_length=1)
            embeddings.append(embedding_layer(rel_input))

        # Concatenate the inputs if there are more than 1.
        if len(embeddings) > 1:
            concatenated_inputs = Concatenate(axis=2)(embeddings)
        else:
            concatenated_inputs = embeddings[0]
        concatenated_inputs = Dropout(self.dropout)(concatenated_inputs)
        return concatenated_inputs


if __name__ == "__main__":
    # This file can be executed to make sure the model compiles.
    args = parse_args()
    rel = ReferringRelationshipsModel(args)
    model = rel.build_model()
    model.summary()
