"""Iterator class that reads the data and batches it for training.
"""

from keras.utils import to_categorical
from keras.utils import Sequence

import argparse
import json
import h5py
import keras.backend as K
import numpy as np
import os

class DiscoveryIterator(Sequence):
    """Discovery version of iterator that drops objects and subjects.
    """

    def __init__(self, data_dir, args):
        """Constructor for the iterator.

        Args:
            data_dir: Location of the annotations.
            args: The arguments from the `config.py` file.
        """
        self.data_dir = data_dir
        self.input_dim = args.input_dim
        self.use_subject = args.use_subject
        self.use_predicate = args.use_predicate
        self.use_object = args.use_object
        self.batch_size = args.batch_size
        self.categorical_predicate = args.categorical_predicate
        self.use_internal_loss = args.use_internal_loss
        self.num_predicates = args.num_predicates
        self.num_objects = args.num_objects

        # Drop variables.
        self.subject_droprate = args.subject_droprate
        self.object_droprate = args.object_droprate
        self.always_drop_file = args.always_drop_file
        if self.always_drop_file is not None:
            self.drop_list = json.load(open(self.always_drop_file))
        else:
            self.drop_list = []

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
            self.subjects = dataset['subject_locations']
            self.objects = dataset['object_locations']

        start_idx = idx * self.batch_size
        end_idx = min(self.samples, (idx + 1) * self.batch_size)

        # Create the batches.
        batch_rel = self.categories[start_idx:end_idx]
        batch_s_regions = self.subjects[start_idx:end_idx].reshape(
            self.batch_size, self.target_size)
        batch_o_regions = self.objects[start_idx:end_idx].reshape(
            self.batch_size, self.target_size)
        current_batch_size = end_idx - start_idx
        batch_image = np.zeros((current_batch_size,) + self.image_shape,
                               dtype=K.floatx())
        for i, image_index in enumerate(batch_rel[:, 3]):
            batch_image[i] = self.images[image_index]

        # Choose the inputs based on the parts of the relationship we will use.
        inputs = [batch_image]
        if self.use_subject:
            subject_masks = np.random.choice(
                    2, end_idx-start_idx,
                    p=[self.subject_droprate, 1.0 - self.subject_droprate,])
            subject_cats = batch_rel[:, 0]
            subject_cats[subject_masks == 0] = self.num_objects
            inputs.append(subject_cats)
        if self.use_predicate:
            if self.categorical_predicate:
                inputs.append(to_categorical(batch_rel[:, 1], num_classes=self.num_predicates))
            else:
                inputs.append(batch_rel[:, 1])
        if self.use_object:
            object_masks = np.random.choice(
                    2, end_idx-start_idx,
                    p=[self.object_droprate, 1.0 - self.object_droprate])
            object_cats = batch_rel[:, 2]
            object_cats[object_masks == 0] = self.num_objects
            inputs.append(object_cats)
        outputs = [batch_s_regions, batch_o_regions]
        return inputs, outputs


class SmartIterator(Sequence):
    """Smart version of iterator that corresponds to `data.SmartDataset`.
    """

    def __init__(self, data_dir, args):
        """Constructor for the iterator.

        Args:
            data_dir: Location of the annotations.
            args: The arguments from the `config.py` file.
        """
        self.data_dir = data_dir
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.use_subject = args.use_subject
        self.use_predicate = args.use_predicate
        self.use_object = args.use_object
        self.batch_size = args.batch_size
        self.categorical_predicate = args.categorical_predicate
        self.use_internal_loss = args.use_internal_loss
        self.num_predicates = args.num_predicates
        # Set the sizes of targets and images.
        self.target_size = self.output_dim * self.output_dim
        self.image_shape = (self.input_dim, self.input_dim, 3)
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
            self.subjects = dataset['subject_locations']
            self.objects = dataset['object_locations']

        start_idx = idx * self.batch_size
        end_idx = min(self.samples, (idx + 1) * self.batch_size)

        # Create the batches.
        batch_rel = self.categories[start_idx:end_idx]
        batch_s_regions = self.subjects[start_idx:end_idx].reshape(
            self.batch_size, self.target_size)
        batch_o_regions = self.objects[start_idx:end_idx].reshape(
            self.batch_size, self.target_size)
        current_batch_size = end_idx - start_idx
        batch_image = np.zeros((current_batch_size,) + self.image_shape,
                               dtype=K.floatx())
        for i, image_index in enumerate(batch_rel[:, 3]):
            batch_image[i] = self.images[image_index]

        # Choose the inputs based on the parts of the relationship we will use.
        inputs = [batch_image]
        if self.use_subject:
            inputs.append(batch_rel[:, 0])
        if self.use_predicate:
            if self.categorical_predicate:
                inputs.append(to_categorical(batch_rel[:, 1], num_classes=self.num_predicates))
            else:
                inputs.append(batch_rel[:, 1])
        if self.use_object:
            inputs.append(batch_rel[:, 2])
        outputs = [batch_s_regions, batch_o_regions]
        return inputs, outputs


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Iterator test.')
    parser.add_argument('--data-dir', type=str,
                        default='data/vrd-10-10-2017/test/',
                        help='Location of the dataset.')
    parser.add_argument('--dataset-type', type=str, default='smart',
                        help='[smart|discovery]')
    parser.add_argument('--input-dim', type=int, default=224,
                        help='Size of the input image.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='The batch size used in training.')
    parser.add_argument('--num-print', type=int, default=1,
                        help='Number of entries to print.')
    args = parser.parse_args()

    args.use_subject = True
    args.use_predicate = True
    args.use_object = True

    dataset_dict = {'smart': SmartIterator, 'discovery': DiscoveryIterator}
    dataset = dataset_dict[args.dataset_type](args.data_dir, args)
    print('Length of dataset: %d' % len(dataset))
    print('Samples in dataset: %d'% dataset.samples)
    count = 0
    for inputs, outputs in dataset:
        print('-'*20)
        print('Image size: %d, %d, %d, %d' % inputs[0].shape)
        print('Image avg pixel: %f' % np.average(inputs[0]))
        print('Subject category: %d' % inputs[1][0])
        print('Predicate category: %d' % inputs[2][0])
        print('Object category: %d' % inputs[3][0])
        print('Average subject heatmap pixels: %f' % np.average(outputs[0]))
        print('Average object heatmap pixels: %f' % np.average(outputs[1]))
        if count >= args.num_print:
            break
        count += 1
