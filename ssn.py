"""Define the referring relationship model.
"""

from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, UpSampling2D, Input, Activation
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.core import Lambda, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.merge import Multiply
from keras.models import Model

from config import parse_args


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

    def build_model(self):
        """Initializes the SSN model.

        Returns:
            The Keras model.
        """
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_subj = Input(shape=(1,))
        input_pred = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        im_features = self.build_image_model(input_im)
        obj_subj_embedding = self.build_embedding_layer(self.num_objects)
        predicate_embedding = self.build_embedding_layer(self.num_predicates)
        subject_features = obj_subj_embedding(input_subj)
        predicate_features = predicate_embedding(input_pred)
        object_features = obj_subj_embedding(input_obj)
        subject_att = self.build_attention_layer(self, im_features, subject_features)
        predicate_att = self.build_attention_layer(self, subject_att, predicate_features)
        object_att = self.build_attention_layer(self, predicate_att, object_features)
        object_regions = self.build_upsampling_layer(object_att, "object_att")
        object_regions_flat = Flatten()(object_regions)
        subject_regions = self.build_upsampling_layer(subject_att, "subject_att")
        subject_regions_flat = Flatten()(subject_regions)
        return [subject_regions_flat, object_regions_flat]

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
        output = base_model.get_layer('activation_40').output
        output = Dense(self.hidden_dim)(output)
        image_branch = Model(inputs=base_model.input, outputs=output)
        im_features = image_branch(input_im)
        return im_features

    def build_embedding_layer(self, num_categories):
        return Embedding(num_categories, self.embedding_dim, input_length=1)

    def build_attention_layer(self, feature_map, query):
        attention_weights = Multiply()([feature_map, query])
        attention_weights = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(attention_weights)
        attention_weights = Activation('sigmoid')(attention_weights)
        attended_map = Multiply()([feature_map, attention_weights])
        return attended_map

    def build_frac_strided_transposed_conv_layer(self, conv_layer):
        res = UpSampling2D(size=(2, 2))(conv_layer)
        res = Conv2DTranspose(1, 3, padding='same')(res)
        return res

    def build_upsampling_layer(self, feature_map, layer_name):
        att = Dense(1, activation='relu')(feature_map)
        upsampled = self.build_frac_strided_transposed_conv_layer(att)
        upsampled = self.build_frac_strided_transposed_conv_layer(upsampled)
        upsampled = self.build_frac_strided_transposed_conv_layer(upsampled)
        upsampled = self.build_frac_strided_transposed_conv_layer(upsampled)
        predictions = Activation('sigmoid', name=layer_name)(upsampled)
        return predictions
    

if __name__ == "__main__":
    args = parse_args()
    rel = ReferringRelationshipsModel(args)
    model = rel.build_model()
    model.summary()
