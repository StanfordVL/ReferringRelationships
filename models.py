"""Defines the shifting attention referring relationship model.
"""

from BilinearUpSampling import BilinearUpSampling2D
from config import parse_args
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, UpSampling2D, Input, Activation, BatchNormalization, RepeatVector, MaxPooling2D
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.core import Lambda, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Multiply, Concatenate
from keras.models import Model
from keras.regularizers import l2
from resnet_helpers import conv_block, identity_block, atrous_conv_block, atrous_identity_block
from utils.visualization_utils import objdict

import numpy as np
import json
import os
import h5py


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
        self.reg = args.reg
        self.use_internal_loss = args.use_internal_loss
        self.internal_loss_weight = args.internal_loss_weight
        self.iterations = args.iterations
        self.baseline_weights = args.baseline_weights
        self.attention_conv_kernel = args.attention_conv_kernel
        self.refinement_conv_kernel = args.refinement_conv_kernel
        self.upsampling_channels = args.upsampling_channels
        self.reg = args.reg
        self.batch_momentum = args.batch_momentum
        self.fcnn = args.fcnn
        self.finetune_cnn = args.finetune_cnn
        self.output_dim = args.output_dim

        # Discovery.
        if args.discovery:
            self.num_objects += 1

    def build_model(self):
        """Chooses which model based on the arguments.
        """
        if self.model == "ssn":
            return self.build_iterative_ssn_model()
        elif self.model == "sym_ssn":
            return self.build_iterative_sym_ssn_model()
        elif self.model == "baseline":
            return self.build_baseline_model()
        elif self.model == "clean":
            return self.build_clean_model()
        elif self.model == "dirty":
            return self.build_dirty_model()
        else:
            raise ValueError('model argument not recognized. '
                             'Model options: ssn, sym_ssn, baseline, '
                             'baseline_no_predicate')

    def build_dirty_model(self):
        # Inputs.
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        if self.use_predicate:
            input_pred = Input(shape=(self.num_predicates,))
            inputs=[input_im, input_subj, input_pred, input_obj]
        else:
            inputs=[input_im, input_subj, input_obj]

        # ALL LAYERS:
        image_conv = Conv2D(self.hidden_dim, self.conv_im_kernel,
                            strides=(1, 1), padding='same', name='image_conv',
                            kernel_initializer='he_normal', activation='relu',
                            data_format='channels_last',
                            bias_initializer='zeros')
        attention_conv = Conv2D(self.hidden_dim, self.attention_conv_kernel,
                                strides=(1, 1), padding='same',
                                kernel_initializer='he_normal',
                                data_format='channels_last',
                                bias_initializer='zeros')
        refinement_conv = Conv2D(self.hidden_dim, self.refinement_conv_kernel,
                                 strides=(1, 1), padding='same',
                                 kernel_initializer='he_normal',
                                 data_format='channels_last',
                                 bias_initializer='zeros')

        # Create the predicate conv layers.
        if self.use_predicate:
            predicate_masks = Reshape((1, 1, self.num_predicates))(input_pred)
            predicate_modules = self.build_conv_modules(basename='conv{}-predicate{}')
            inverse_predicate_modules = self.build_conv_modules(basename='conv{}-inv-predicate{}')

                # Embeddings
        embedding_layer = Embedding(self.num_objects, self.hidden_dim)
        embedded_subject = embedding_layer(input_subj)
        embedded_object = embedding_layer(input_obj)

        # Image features
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(self.input_dim, self.input_dim, 3))
        for layer in base_model.layers:
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
        im_features = image_conv(im_features)

        # Attention
        subject_att = self.attend(im_features, embedded_subject, attention_conv,
                                  name='subject-att-0')
        object_att = self.attend(im_features, embedded_object, attention_conv,
                                 name='object-att-0')

        if self.use_internal_loss:
            subject_outputs = [subject_att]
            object_outputs = [object_att]

        # Iterate!
        for iteration in range(self.iterations):
            predicate_att = self.transform_conv_attention(
                subject_att, predicate_modules, predicate_masks)
            predicate_att = Lambda(
                lambda x: x, name='shift-{}'.format(iteration+1))(predicate_att)
            new_image_features = self.refine_features(
                im_features, predicate_att, refinement_conv)
            new_object_att = self.attend(
                new_image_features, embedded_object, attention_conv,
                name='object-att-{}'.format(iteration+1))

            inv_predicate_att = self.transform_conv_attention(
                object_att, inverse_predicate_modules, predicate_masks)
            inv_predicate_att = Lambda(
                lambda x: x, name='inv-shift-{}'.format(iteration+1))(inv_predicate_att)
            new_image_features = self.refine_features(
                im_features, inv_predicate_att, refinement_conv)
            new_subject_att = self.attend(
                new_image_features, embedded_subject, attention_conv,
                name='subject-att-{}'.format(iteration+1))

            if self.use_internal_loss:
                object_outputs.append(new_object_att)
                subject_outputs.append(new_subject_att)
            object_att = new_object_att
            subject_att = new_subject_att

        if self.use_internal_loss and self.iterations > 0:
            subject_att = Concatenate(axis=3)(subject_outputs)
            object_att = Concatenate(axis=3)(object_outputs)
            subject_att = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(subject_att)
            object_att = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(object_att)


        # outputting
        subject_regions = Reshape((self.output_dim * self.output_dim,),
                                  name='subject')(subject_att)
        object_regions = Reshape((self.output_dim * self.output_dim,),
                                 name='object')(object_att)
        model = Model(inputs=inputs, outputs=[subject_regions, object_regions])
        return model

    def refine_features(self, im_features, predicate_att, refinement_conv):
        shifted_features = Multiply()([predicate_att, im_features])
        combined_features = Concatenate(axis=3)([im_features, shifted_features])
        new_image_features = refinement_conv(combined_features)
        return new_image_features

    def build_clean_model(self):
        # ALL LAYERS:
        image_conv = Conv2D(self.hidden_dim, self.conv_im_kernel,
                            strides=(1, 1), padding='same', name='image_conv',
                            kernel_initializer='he_normal', activation='relu',
                            data_format='channels_last',
                            bias_initializer='zeros')
        attention_conv = Conv2D(self.hidden_dim, self.attention_conv_kernel,
                                strides=(1, 1), padding='same',
                                kernel_initializer='he_normal',
                                data_format='channels_last',
                                bias_initializer='zeros')

        # Inputs.
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        if self.use_predicate:
            input_pred = Input(shape=(self.num_predicates,))
            inputs=[input_im, input_subj, input_pred, input_obj]
        else:
            inputs=[input_im, input_subj, input_obj]

        # Embeddings
        embedding_layer = Embedding(self.num_objects, self.hidden_dim)
        embedded_subject = embedding_layer(input_subj)
        embedded_object = embedding_layer(input_obj)

        # Image features
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(self.input_dim, self.input_dim, 3))
        for layer in base_model.layers:
            layer.trainable = False
            layer.training = False
        output = base_model.get_layer(self.feat_map_layer).output
        image_branch = Model(inputs=base_model.input, outputs=output)
        im_features = image_branch(input_im)
        im_features = image_conv(im_features)

        # Attention
        subject_att = self.attend(im_features, embedded_subject, attention_conv,
                                  name='subject-att-0')
        object_att = self.attend(im_features, embedded_object, attention_conv,
                                 name='object-att-0')

        # outputting
        subject_regions = Reshape((self.output_dim * self.output_dim,),
                                  name='subject')(subject_att)
        object_regions = Reshape((self.output_dim * self.output_dim,),
                                 name='object')(object_att)
        model = Model(inputs=inputs, outputs=[subject_regions, object_regions])
        return model

    def upsample_clean(att, upsamplers, conv_transposes):
        for upsample_layer, conv_transpose_layer in zip(upsamplers,
                                                        conv_transposes):
            att = upsample_layer(att)
            att = conv_transpose_layer(att)
        return att

    def build_custom_resnet_features(self, input_im):
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1',
                   kernel_regularizer=l2(self.reg))(input_im)
        x = BatchNormalization(axis=3, name='bn_conv1',
                               momentum=self.batch_momentum)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(3, [64, 64, 256], stage=2, block='a',
                       weight_decay=self.reg, strides=(1, 1),
                       batch_momentum=self.batch_momentum)(x)
        x = identity_block(3, [64, 64, 256], stage=2, block='b',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)
        x = identity_block(3, [64, 64, 256], stage=2, block='c',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)

        x = conv_block(3, [128, 128, 512], stage=3, block='a',
                       weight_decay=self.reg,
                       batch_momentum=self.batch_momentum)(x)
        x = identity_block(3, [128, 128, 512], stage=3, block='b',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)
        x = identity_block(3, [128, 128, 512], stage=3, block='c',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)
        x = identity_block(3, [128, 128, 512], stage=3, block='d',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)

        x = conv_block(3, [256, 256, 1024], stage=4, block='a',
                       weight_decay=self.reg,
                       batch_momentum=self.batch_momentum)(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='b',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='c',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='d',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)
        x = identity_block(3, [256, 256, 1024], stage=4, block='e',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)
        im_features = identity_block(3, [256, 256, 1024], stage=4, block='f',
                           weight_decay=self.reg,
                           batch_momentum=self.batch_momentum)(x)

        return im_features

    def build_fcnn_upsampling_layer(self):
        b1 = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a',
                              weight_decay=self.reg,
                              atrous_rate=(2, 2),
                              batch_momentum=self.batch_momentum)
        b2 = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b',
                                  weight_decay=self.reg,
                                  atrous_rate=(2, 2),
                                  batch_momentum=self.batch_momentum)
        b3 = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c',
                                  weight_decay=self.reg,
                                  atrous_rate=(2, 2),
                                  batch_momentum=self.batch_momentum)
        b4 = Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='linear',
                   padding='same', strides=(1, 1),
                   kernel_regularizer=l2(self.reg))
        b5 = BilinearUpSampling2D(
            target_size=(self.input_dim, self.input_dim))
        return [b1, b2, b3, b4, b5]

    def load_fcnn_weights(self, model):
        weights_path = os.path.join(
            '/data/chami/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5')
        model.load_weights(weights_path, by_name=True)

    def build_iterative_sym_ssn_model(self):
        """Iteratives build focusing on the subject and object over and over again.

        Returns:
            The Keras model.
        """
        # Inputs.
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        if self.use_predicate:
            input_pred = Input(shape=(self.num_predicates,))
            inputs=[input_im, input_subj, input_pred, input_obj]
        else:
            inputs=[input_im, input_subj, input_obj]

        # Extract image features and embeddings
        if self.baseline_weights:
            params = objdict(json.load(open(os.path.join(os.path.dirname(self.baseline_weights), "args.json"), "r")))
            params.baseline_weights = None
            model_weights = h5py.File(self.baseline_weights)
            relationships_model = ReferringRelationshipsModel(params)
            base_model = relationships_model.build_model()
            base_model.load_weights(self.baseline_weights)
            output_im = base_model.get_layer("conv2d_1").output
            output_emb_s = base_model.get_layer("activation_50").output
            output_emb_o = base_model.get_layer("activation_51").output
            baseline_branch = Model(inputs=base_model.input, outputs=[output_im, output_emb_s, output_emb_o])
            for layer in baseline_branch.layers:
                layer.trainable = False
            im_features, embedded_subject, embedded_object = baseline_branch([input_im, input_subj, input_obj])
        else:
            if self.fcnn:
                im_features = self.build_custom_resnet_features(input_im)
                upsampling_layers = self.build_fcnn_upsampling_layer()
            else:
                im_features = self.build_image_model(input_im)
            subj_obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
            embedded_subject = subj_obj_embedding(input_subj)
            embedded_subject = Activation("relu")(embedded_subject)
            embedded_subject = Dropout(self.dropout)(embedded_subject)
            embedded_object = subj_obj_embedding(input_obj)
            embedded_object =  Activation("relu")(embedded_object)
            embedded_object = Dropout(self.dropout)(embedded_object)

        # Refinement parameters
        refinement_conv = Conv2D(self.hidden_dim, self.refinement_conv_kernel, strides=(1, 1), padding='same')
        attention_conv = Conv2D(self.hidden_dim, self.attention_conv_kernel, strides=(1, 1), padding='same')

        # Extract initial attention maps.
        subject_att = self.attend(im_features, embedded_subject, attention_conv, name='subject-att-0')
        object_att = self.attend(im_features, embedded_object, attention_conv, name='object-att-0')

        # Create the predicate conv layers.
        if self.use_predicate:
            predicate_masks = Reshape((1, 1, self.num_predicates))(input_pred)
            predicate_modules = self.build_conv_modules(basename='conv{}-predicate{}')
            inverse_predicate_modules = self.build_conv_modules(basename='conv{}-inv-predicate{}')

        if self.use_internal_loss:
            subject_outputs = [subject_att]
            object_outputs = [object_att]

        # Iterate!
        for iteration in range(self.iterations):
            predicate_att = self.transform_conv_attention(subject_att, predicate_modules, predicate_masks)
            predicate_att = Lambda(lambda x: x, name='shift-{}'.format(iteration+1))(predicate_att)
            new_image_features = Multiply()([im_features, predicate_att])
            new_image_features = Concatenate(axis=3)([im_features, new_image_features])
            new_image_features = refinement_conv(new_image_features)
            new_object_att = self.attend(new_image_features, embedded_object, attention_conv, name='object-att-{}'.format(iteration+1))

            inv_predicate_att = self.transform_conv_attention(object_att, inverse_predicate_modules, predicate_masks)
            inv_predicate_att = Lambda(lambda x: x, name='inv-shift-{}'.format(iteration+1))(inv_predicate_att)
            new_image_features = Multiply()([im_features, inv_predicate_att])
            new_image_features = Concatenate(axis=3)([im_features, new_image_features])
            new_image_features = refinement_conv(new_image_features)
            new_subject_att = self.attend(new_image_features, embedded_subject, attention_conv, name='subject-att-{}'.format(iteration+1))

            if self.use_internal_loss:
                object_outputs.append(new_object_att)
                subject_outputs.append(new_subject_att)
            object_att = new_object_att
            subject_att = new_subject_att

        # Upsample the subject and objects regions.
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
            subject_att = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(subject_att)
            object_att = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(object_att)

        if self.fcnn:
            subject_regions = subject_att
            object_regions = object_att
            for layer in upsampling_layers:
                subject_regions = layer(subject_regions)
                object_regions = layer(object_regions)
            subject_regions = Reshape((self.input_dim * self.input_dim,), name='subject')(subject_regions)
            object_regions = Reshape((self.input_dim * self.input_dim,), name='object')(object_regions)
        else:
            object_regions = self.build_upsampling_layer(object_att, name="object")
            subject_regions = self.build_upsampling_layer(subject_att, name="subject")

        # Create and output the model.
        model = Model(inputs=inputs, outputs=[subject_regions, object_regions])
        if self.fcnn:
            self.load_fcnn_weights(model)
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
            for i in range(self.nb_conv_att_map):
                channels = self.conv_predicate_channels
                if i == self.nb_conv_att_map-1:
                    # last conv with only one channel
                    channels = 1
                predicate_conv = Conv2D(
                    channels, self.conv_predicate_kernel,
                    strides=(1, 1), padding='same', use_bias=False,
                    kernel_initializer='he_normal', activation='relu',
                    data_format='channels_last',
                    name=basename.format(i, k))
                predicate_module_group.append(predicate_conv)
            predicate_modules.append(predicate_module_group)
        return predicate_modules

    def build_iterative_ssn_model(self):
        """Iteratives build focusing on the subject and object over and over again.

        Returns:
            The Keras model.
        """
        # Inputs.
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_pred = Input(shape=(self.num_predicates,))
        input_obj = Input(shape=(1,))

        # Extract image features.
        im_features = self.build_image_model(input_im)

        # Extract object embeddings.
        subj_obj_embedding = self.build_embedding_layer(self.num_objects, self.hidden_dim)
        embedded_subject = subj_obj_embedding(input_subj)
        embedded_subject = Activation('relu')(embedded_subject)
        embedded_subject = Dropout(self.dropout)(embedded_subject)
        embedded_object = subj_obj_embedding(input_obj)
        embedded_object = Activation('relu')(embedded_object)

        # Extract subject attention map.
        subject_att = self.attend(im_features, embedded_subject, name='subject-att-0')

        # Create the predicate conv layers.
        predicate_masks = Reshape((1, 1, self.num_predicates))(input_pred)
        predicate_modules = self.build_conv_modules(basename='conv{}-predicate{}')
        inverse_predicate_modules = self.build_conv_modules(basename='conv{}-inv-predicate{}')

        # Iterate!
        if self.use_internal_loss:
            subject_outputs = [subject_att]
            object_outputs = []
        for iteration in range(self.iterations):
            if iteration % 2 == 0:
                predicate_att = self.transform_conv_attention(
                    subject_att, predicate_modules, predicate_masks)
                predicate_att = Lambda(lambda x: x, name='shift-{}'.format(iteration+1))(
                    predicate_att)
                new_im_features = Multiply()([im_features, predicate_att])
                object_att = self.attend(new_im_features, embedded_object,
                                         name='object-att-{}'.format(iteration+1))
                if self.use_internal_loss:
                    object_outputs.append(object_att)
            else:
                predicate_att = self.transform_conv_attention(
                    object_att, inverse_predicate_modules, predicate_masks)
                predicate_att = Lambda(lambda x: x, name='inv-shift-{}'.format(
                    iteration+1))(predicate_att)
                new_im_features = Multiply()([im_features, predicate_att])
                subject_att = self.attend(new_im_features, embedded_subject,
                                          name='subject-att-{}'.format(iteration+1))
                if self.use_internal_loss:
                    subject_outputs.append(subject_att)

        # Combine all the internal subject predictions..
        if self.use_internal_loss and len(subject_outputs) > 1:
            internal_subject_weights = np.array([self.internal_loss_weight**iteration for iteration in range(len(subject_outputs))])
            internal_subject_weights = internal_subject_weights/internal_subject_weights.sum()
            # Multiply with the internal losses.
            subject_outputs = Lambda(lambda x: [internal_subject_weights[i]*x[i] for i in range(len(x))])(subject_outputs)
            # Concatenate all the internal outputs.
            subject_att = Concatenate(axis=3)(subject_outputs)
            # Sum across the internal values.
            subject_att = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(subject_att)

        # Combine all the internal object predictions.
        if self.use_internal_loss and len(object_outputs) > 1:
            internal_object_weights = np.array([self.internal_loss_weight**iteration for iteration in range(len(object_outputs))])
            internal_object_weights = internal_object_weights/internal_object_weights.sum()
            # Multiply with the internal losses.
            object_outputs = Lambda(lambda x: [internal_object_weights[i]*x[i] for i in range(len(x))])(object_outputs)
            # Concatenate all the internal outputs.
            object_att = Concatenate(axis=3)(object_outputs)
            # Sum across the internal values.
            object_att = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(object_att)

        # Upsample the subject regions.
        subject_regions = self.build_upsampling_layer(subject_att, name="subject")
        # Upsample the object regions.
        object_regions = self.build_upsampling_layer(object_att, name="object")

        # Create and output the model.
        model = Model(inputs=[input_im, input_subj, input_pred, input_obj], outputs=[subject_regions, object_regions])
        return model


    def transform_conv_attention(self, att, merged_modules, predicate_masks):
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
        shapify = Reshape((self.feat_map_dim, self.feat_map_dim, 1, 1))
        conv_outputs = []
        for group in merged_modules:
            att_map = att
            for conv_module in group:
                att_map = conv_module(att_map)
            att_map = shapify(att_map)
            conv_outputs.append(att_map)
        merged_output = Concatenate(axis=4)(conv_outputs)
        predicate_att = Multiply()([predicate_masks, merged_output])
        predicate_att = Lambda(lambda x: K.sum(x, axis=4))(predicate_att)
        predicate_att = Activation("tanh")(predicate_att)
        return predicate_att

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

        # Refinement parameters
        attention_conv = Conv2D(self.hidden_dim, self.attention_conv_kernel, strides=(1, 1), padding='same')

        # Map the inputs to the outputs.
        im_features = self.build_image_model(input_im)
        rel_features = self.build_relationship_model(relationship_inputs, num_classes)
        rel_features = Dropout(self.dropout)(rel_features)
        subjects_features = Dense(self.hidden_dim)(rel_features)
        objects_features = Dense(self.hidden_dim)(rel_features)
        subjects_features = Dropout(self.dropout)(subjects_features)
        objects_features = Dropout(self.dropout)(objects_features)
        subject_att = self.attend(im_features, subjects_features, attention_conv)
        object_att = self.attend(im_features, objects_features, attention_conv)
        subject_regions = self.build_upsampling_layer(subject_att, name="subject")
        object_regions = self.build_upsampling_layer(object_att, name="object")
        model_inputs = [input_im] + relationship_inputs
        model = Model(inputs=model_inputs, outputs=[subject_regions, object_regions])
        return model

    def build_image_model(self, input_im):
        """Grab the image features.

        Args:
            input_im: The input image to the model.

        Returns:
            The image feature map.
        """
        if self.cnn == "resnet":
            base_model = ResNet50(weights='imagenet',
                                  include_top=False,
                                  input_shape=(self.input_dim, self.input_dim, 3))
        elif self.cnn == 'vgg':
            base_model = VGG19(weights='imagenet',
                               include_top=False,
                               input_shape=(self.input_dim, self.input_dim, 3))
        else:
            raise ValueError('--cnn parameter not recognized.')
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
        return Embedding(num_categories, emb_dim, input_length=1)

    def attend_add(self, feature_map, query, conv_op, name=None):
        query = Reshape((self.hidden_dim,))(query)
        query = RepeatVector(self.feat_map_dim)(query)
        query = Reshape((self.feat_map_dim * self.hidden_dim,))(query)
        query = RepeatVector(self.feat_map_dim)(query)
        query = Reshape((self.feat_map_dim, self.feat_map_dim, self.hidden_dim))(query)
        attention_weights = Concatenate(axis=3)([feature_map, query])
        attention_weights = conv_op(attention_weights)
        attention_weights = Activation("relu", name=name)(attention_weights)
        return attention_weights

    def attend_refine(self, feature_map, query, conv_op, name=None):
        query = Reshape((self.hidden_dim,))(query)
        query = RepeatVector(self.feat_map_dim)(query)
        query = Reshape((self.feat_map_dim * self.hidden_dim,))(query)
        query = RepeatVector(self.feat_map_dim)(query)
        query = Reshape((self.feat_map_dim, self.feat_map_dim, self.hidden_dim))(query)
        attention_weights = Concatenate(axis=3)([feature_map, query])
        attention_weights = conv_op(attention_weights)
        attention_weights = Activation("relu", name=name)(attention_weights)
        return attention_weights

    def attend(self, feature_map, query, attention_conv, name=None):
        query = Reshape((1, 1, self.hidden_dim,))(query)
        attention_weights = Multiply()([feature_map, query])
        attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(attention_weights)
        attention_weights = Activation("relu", name=name)(attention_weights)
        attention_weights = Activation("tanh", name=name)(attention_weights)
        return attention_weights

    def build_upsampling_layer(self, feature_map, name=None):
        upsampling_factor = self.input_dim / self.feat_map_dim
        k = int(np.log(upsampling_factor) / np.log(2))
        res = feature_map
        for i in range(k):
            res = UpSampling2D(size=(2, 2), name=name+"-upsampling-{}".format(i))(res)
            num_channels = self.upsampling_channels
            if i == k-1:
                num_channels = 1 # Last output is 1D heatmap.
            res = Conv2DTranspose(num_channels, 3, padding='same', use_bias=False, name=name+"-convT-{}".format(i), activation="relu")(res)
        res = Reshape((self.input_dim * self.input_dim,))(res)
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
