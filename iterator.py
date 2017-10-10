"""Iterator class that reads the data and batches it for training.
"""

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils.data_utils import Sequence

import h5py
import os
import keras.backend as K
import numpy as np


class DatasetIterator(Sequence):
    """Extends Keras backend implementation of an Iterator.
    """

    def __init__(self, data_dir, args, shuffle=True):
        """Constructor for the iterator.

        Args:
            data_dir: Location of the annotations.
            args: The arguments from the `config.py` file.
            shuffle: Boolean deciding whether we should be shuffling the
              data when sampling for the next batch.
        """
        self.data_dir = data_dir
        self.input_dim = args.input_dim
        self.use_subject = args.use_subject
        self.use_predicate = args.use_predicate
        self.use_object = args.use_object
        self.batch_size = args.batch_size

        # Set the sizes of targets and images.
        self.target_size = args.input_dim * args.input_dim
        self.image_shape = (args.input_dim, args.input_dim, 3)
        self.data_format = K.set_image_data_format('channels_last')

        # Load the dataset
        dataset = h5py.File(os.path.join(self.data_dir, 'dataset.hdf5'), 'r')
        self.images = dataset['images']
        self.categories = dataset['categories']
        self.subjects = dataset['subject_locations']
        self.objects = dataset['object_locations']
        self.samples = self.images.shape[0]
        self.length = float(self.samples) /  self.batch_size

    def __len__(self):
        """The number of items in the dataset.

        Returns:
            The number of items in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """Grab the next batch of data for training.

        Args:
            idx: The index between 0 and __len__().

        Returns:
            The next batch as a tuple containing two elements. The first element
            is batch of inputs which contains the image, subject, predicate,
            object. These inputs change depending on which inputs are training
            with. The second element of the tuple contains the output masks we
            want the model to predict.
        """
        start_idx = idx * self.batch_size
        end_idx = min(self.samples, (idx + 1) * self.batch_size)

        # Create the batches.
        batch_image = self.images[start_idx:end_idx]
        batch_rel = self.categories[start_idx:end_idx]
        batch_s_regions = self.subjects[start_idx:end_idx]
        batch_o_regions = self.objects[start_idx:end_idx]

        # Choose the inputs based on the parts of the relationship we will use.
        inputs = [batch_image]
        if self.use_subject:
            inputs.append(batch_rel[:, 0])
        if self.use_predicate:
            inputs.append(batch_rel[:, 1])
        if self.use_object:
            inputs.append(batch_rel[:, 2])
        return inputs, [batch_s_regions, batch_o_regions]


class RefRelDataIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    """

    def __init__(self, data_dir, args, shuffle=True):
        """Constructor for the iterator.

        Args:
            data_dir: Location of the annotations.
            args: The arguments from the `config.py` file.
            shuffle: Boolean deciding whether we should be shuffling the
              data when sampling for the next batch.
        """
        self.data_dir = data_dir
        self.input_dim = args.input_dim
        self.use_subject = args.use_subject
        self.use_predicate = args.use_predicate
        self.use_object = args.use_object

        # Set the sizes of targets and images.
        self.target_size = args.input_dim * args.input_dim
        self.image_shape = (args.input_dim, args.input_dim, 3)
        self.data_format = K.set_image_data_format('channels_last')

        # Load the dataset
        dataset = h5py.File(os.path.join(self.data_dir, 'dataset.hdf5'), 'r')
        self.images = dataset['images']
        self.categories = dataset['categories']
        self.subjects = dataset['subject_locations']
        self.objects = dataset['object_locations']
        self.samples = self.images.shape[0]
        super(RefRelDataIterator, self).__init__(
            self.samples, args.batch_size, shuffle, args.seed)

    def next(self):
        """Grab the next batch of data for training.

        Returns:
            The next batch as a tuple containing two elements. The first element
            is batch of inputs which contains the image, subject, predicate,
            object. These inputs change depending on which inputs are training
            with. The second element of the tuple contains the output masks we
            want the model to predict.
        """
        # Grab the indices for this batch.
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)

        # Create the batches.
        batch_image = np.zeros((current_batch_size,) + self.image_shape,
                               dtype=K.floatx())
        batch_rel = np.zeros((current_batch_size,) + (3,), dtype=K.floatx())
        batch_s_regions = np.zeros((current_batch_size,) + (self.target_size,),
                                   dtype=K.floatx())
        batch_o_regions = np.zeros((current_batch_size,) + (self.target_size,),
                                   dtype=K.floatx())

        # build batch of image data.
        for i, j in enumerate(index_array):
            batch_image[i] = self.images[j]
            batch_rel[i] = self.categories[j]
            batch_s_regions[i] = self.subjects[j].flatten()
            batch_o_regions[i] = self.objects[j].flatten()

        # Choose the inputs based on the parts of the relationship we will use.
        inputs = [batch_image]
        if self.use_subject:
            inputs.append(batch_rel[:, 0])
        if self.use_predicate:
            inputs.append(batch_rel[:, 1])
        if self.use_object:
            inputs.append(batch_rel[:, 2])
        return inputs, [batch_s_regions, batch_o_regions]
