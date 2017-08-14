from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, UpSampling2D, Reshape, Input, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import Dot, Concatenate, Multiply
from keras.layers.core import Lambda
from keras.models import Model, Sequential
from keras import backend as K


class ReferringRelationshipsModel():
    def __init__(self, num_objects, num_subjects, num_predicates, input_dim=224, feat_map_dim=14, hidden_dim=200, embedding_dim=100):
        self.input_dim = input_dim
        self.feat_map_dim = feat_map_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_subjects = num_subjects
        self.num_predicates = num_predicates
        self.num_objects = num_objects
        self.upsampling_factor = input_dim / feat_map_dim

        # ****************************************** MODEL ******************************************

    def build_model(self):
        input_im = Input(shape=(self.input_dim, self.input_dim, 3))
        input_rel = Input(shape=(1,))
        input_obj = Input(shape=(1,))
        input_subj = Input(shape=(1,))
        images = self.build_image_model()(input_im)
        relationships = self.build_relationship_model(input_subj, input_rel, input_obj)
        subject_regions = self.build_attention_layer_2(images, relationships)
        object_regions = self.build_attention_layer_2(images, relationships)
        model = Model(inputs=[input_im, input_subj, input_rel, input_obj], outputs=[subject_regions, object_regions])
        return model

    # *************************************** FLATTEN MODEL ***************************************
    # this is to make sure we preserve the ordering
    def build_flatten_model(self, input_dim):
        flatten_model = Sequential()
        flatten_model.add(Flatten(input_shape=(input_dim, input_dim, 1)))
        return flatten_model

    # *************************************** IMAGE BRANCH ***************************************
    def build_image_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.input_dim, self.input_dim, 3))
        for layer in base_model.layers:
            layer.trainable = False
        output = base_model.get_layer('block5_conv3').output
        output = Dense(self.hidden_dim)(output)
        image_branch = Model(inputs=base_model.input, outputs=output)
        return image_branch

    # ************************************ RELATIONSHIP BRANCH ***********************************
    def build_embedding_layer(self, input_vector, num_categories):
        embedding = Embedding(num_categories, self.embedding_dim, input_length=1)(input_vector)
        return embedding

    def build_relationship_model(self, input_subj, input_rel, input_obj):
        subj_embedding = self.build_embedding_layer(input_subj, self.num_subjects)
        predicate_embedding = self.build_embedding_layer(input_rel, self.num_predicates)
        obj_embedding = self.build_embedding_layer(input_obj, self.num_objects)
        concatenated_inputs = Concatenate(axis=2)([subj_embedding, predicate_embedding, obj_embedding])
        rel_repr = Dense(self.hidden_dim, activation='relu')(concatenated_inputs)
        # rel_repr = Dropout(0.2)(rel_repr)
        return rel_repr

    def build_attention_layer_1(self, images, relationships):
        images = Reshape(target_shape=(self.feat_map_dim * self.feat_map_dim, self.hidden_dim))(images)  # (196)x100
        merged = Dot(axes=(2, 2))([images, relationships])
        reshaped = Reshape(target_shape=(self.feat_map_dim, self.feat_map_dim, 1))(merged)
        upsampled = UpSampling2D(size=(self.upsampling_factor, self.upsampling_factor))(reshaped)
        flattened = Flatten(input_shape=(self.input_dim, self.input_dim, 10))(upsampled)  # todo: what is this 10??
        predictions = Activation('sigmoid')(flattened)
        return predictions

    def build_attention_layer_2(self, images, relationships):
        merged = Multiply()([images, relationships])
        merged = Lambda(lambda x: K.sum(x, axis=3))(merged)
        merged = Reshape(target_shape=(self.feat_map_dim, self.feat_map_dim, 1))(merged)
        upsampled = UpSampling2D(size=(self.upsampling_factor, self.upsampling_factor))(merged)
        flattened = Flatten(input_shape=(self.input_dim, self.input_dim, 10))(upsampled)  # todo: what is this 10??
        predictions = Activation('sigmoid')(flattened)
        return predictions

if __name__ == "__main__":
    rel = ReferringRelationshipsModel(num_objects=10, num_subjects=10, num_predicates=10)
    model = rel.build_model()
    model.summary()
