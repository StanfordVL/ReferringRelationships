"""Iterator class that reads the data and batches it for training.
"""

from keras.preprocessing.image import Iterator
from keras.utils import Sequence, to_categorical

import h5py
import keras.backend as K
import math
import numpy as np
import os

class SemanticSegmentationIterator(Sequence):
    """Iterator for pascal.
    """

    def __init__(self, data_dir, args):
        """Constructor for the iterator.

        Args:
            data_dir: Location of the annotations.
            args: The arguments from the `config.py` file.
        """
        self.data_dir = data_dir
        self.input_dim = args.input_dim
        self.batch_size = args.batch_size
        self.num_objets = args.num_objects

        # Set the sizes of targets and images.
        self.target_size = args.input_dim * args.input_dim * args.num_objects
        self.image_shape = (args.input_dim, args.input_dim, 3)
        self.data_format = K.set_image_data_format('channels_last')

        # Load the dataset
        dataset = h5py.File(os.path.join(self.data_dir, 'dataset.hdf5'), 'r')
        objects = dataset['object_locations']
        self.samples = objects.shape[0]
        self.length = int(float(self.samples) /  self.batch_size)

    def __len__(self):
        """The number of items in the dataset.

        Returns:
            The number of items in the dataset.
        """
        return self.length

    def get_image_dataset(self):
        """Retrieves the image dataset.

        Returns:
            The image hdf5 dataset.
        """
        dataset = h5py.File(os.path.join(self.data_dir, 'images.hdf5'), 'r')
        return dataset['images']

    def on_epoch_end(self):
        return

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
        if not hasattr(self, 'images'):
            images = h5py.File(os.path.join(self.data_dir, 'images.hdf5'), 'r')
            dataset = h5py.File(os.path.join(self.data_dir, 'dataset.hdf5'), 'r')
            self.images = images['images']
            self.objects = dataset['object_locations']

        start_idx = idx * self.batch_size
        end_idx = min(self.samples, (idx + 1) * self.batch_size)

        # Create the batches.
        batch_o_regions = self.objects[start_idx:end_idx].reshape(
            self.batch_size, self.target_size)
        current_batch_size = end_idx - start_idx
        batch_image = self.images[start_idx:end_idx]

        # Choose the inputs based on the parts of the relationship we will use.
        inputs = [batch_image]
        outputs = [batch_o_regions]
        return inputs, outputs


class CLassSegmentationIterator(Sequence):
    """Iterator for pascal.
    """

    def __init__(self, data_dir, args):
        """Constructor for the iterator.

        Args:
            data_dir: Location of the annotations.
            args: The arguments from the `config.py` file.
        """
        self.data_dir = data_dir
        self.input_dim = args.input_dim
        self.batch_size = args.batch_size

        # Set the sizes of targets and images.
        self.target_size = args.input_dim * args.input_dim
        self.image_shape = (args.input_dim, args.input_dim, 3)
        self.data_format = K.set_image_data_format('channels_last')

        # Load the dataset
        dataset = h5py.File(os.path.join(self.data_dir, 'dataset.hdf5'), 'r')
        categories = dataset['categories']
        self.samples = categories.shape[0]
        self.length = int(float(self.samples) /  self.batch_size)

    def __len__(self):
        """The number of items in the dataset.

        Returns:
            The number of items in the dataset.
        """
        return self.length

    def get_image_dataset(self):
        """Retrieves the image dataset.

        Returns:
            The image hdf5 dataset.
        """
        dataset = h5py.File(os.path.join(self.data_dir, 'images.hdf5'), 'r')
        return dataset['images']

    def on_epoch_end(self):
        return

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
        if not hasattr(self, 'images'):
            images = h5py.File(os.path.join(self.data_dir, 'images.hdf5'), 'r')
            dataset = h5py.File(os.path.join(self.data_dir, 'dataset.hdf5'), 'r')
            self.images = images['images']
            self.categories = dataset['categories']
            self.objects = dataset['object_locations']

        start_idx = idx * self.batch_size
        end_idx = min(self.samples, (idx + 1) * self.batch_size)

        # Create the batches.
        batch_rel = self.categories[start_idx:end_idx]
        batch_o_regions = self.objects[start_idx:end_idx].reshape(
            self.batch_size, self.target_size)
        current_batch_size = end_idx - start_idx
        batch_image = np.zeros((current_batch_size,) + self.image_shape,
                               dtype=K.floatx())
        for i, image_index in enumerate(batch_rel[:, 1]):
            batch_image[i] = self.images[image_index]

        # Choose the inputs based on the parts of the relationship we will use.
        inputs = [batch_image, batch_rel[:, 0]]
        outputs = [batch_o_regions]
        return inputs, outputs
