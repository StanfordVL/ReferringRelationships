import numpy as np
import os
import sys
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.models import model_from_json
from keras.utils import np_utils
from keras.applications.vgg16 import *
from keras.applications.resnet50 import *
from keras.utils.data_utils import get_file
import keras.backend as K
import tensorflow as tf

from resnet_helpers import *


def get_weights_path_vgg16():
    TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',TF_WEIGHTS_PATH,cache_subdir='models')
    return weights_path

def get_weights_path_resnet():
    TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',TF_WEIGHTS_PATH,cache_subdir='models')
    return weights_path

def transfer_FCN_ResNet50():
    input_shape = (224, 224, 3)
    img_input = Input(shape=input_shape)
    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

    x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)

    x = Conv2D(1000, (1, 1), activation='linear', name='fc1000')(x)

    # Create model
    model = Model(img_input, x)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))

    #transfer if weights have not been created
    if os.path.isfile(weights_path) == False:
        flattened_layers = model.layers
        index = {}
        for layer in flattened_layers:
            if layer.name:
                index[layer.name]=layer
        resnet50 = ResNet50()
        for layer in resnet50.layers:
            weights = layer.get_weights()
            if layer.name=='fc1000':
                weights[0] = np.reshape(weights[0], (1,1,2048,1000))
            if layer.name in list(index.keys()):
                index[layer.name].set_weights(weights)
        model.save_weights(weights_path)
        print( 'Successfully transformed!')
    #else load weights
    else:
        model.load_weights(weights_path, by_name=True)
        print( 'Already transformed!')

if __name__ == '__main__':
    func = globals()['transfer_FCN_%s'%sys.argv[1]]
    func()
