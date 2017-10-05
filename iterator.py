"""Iterator class that reads the data and batches it for training.
"""

from keras.preprocessing.image import Iterator, load_img, img_to_array

import os
import keras.backend as K
import numpy as np


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
        self.image_dir = args.image_data_dir
        self.input_dim = args.input_dim
        self.use_subject = args.use_subject
        self.use_predicate = args.use_predicate
        self.use_object = args.use_object
        self.target_size = (args.input_dim, args.input_dim)
        self.data_format = K.image_data_format()
        if self.data_format == 'channels_last':
            self.rgb_image_shape = self.target_size + (3,)
        else:
            self.rgb_image_shape = (3,) + self.target_size
        self.rel_idx = np.load(os.path.join(self.data_dir, 'rel_idx.npy'))
        self.relationships = np.load(os.path.join(self.data_dir,
                                                  'relationships.npy'))
        self.samples = self.rel_idx.shape[0]
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
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        batch_image = np.zeros((current_batch_size,) + self.rgb_image_shape,
                               dtype=K.floatx())
        batch_rel = np.zeros((current_batch_size,) + (3,), dtype=K.floatx())
        im_size = self.input_dim * self.input_dim
        batch_s_regions = np.zeros((current_batch_size,) + (im_size,),
                                   dtype=K.floatx())
        batch_o_regions = np.zeros((current_batch_size,) + (im_size,),
                                   dtype=K.floatx())

        # build batch of image data.
        for i, j in enumerate(index_array):
            rel_id = self.rel_idx[j]
            rel = self.relationships[j]
            image_fname = ''.join(rel_id.split('-')[:-1])
            subject_fname = rel_id + '-s.jpg'
            object_fname = rel_id + '-o.jpg'

            # Load the image.
            try:
                img_path = os.path.join(self.image_dir, image_fname + '.jpg')
                img = load_img(img_path, grayscale=False,
                               target_size=self.target_size)
            except IOError:
                img_path = os.path.join(self.image_dir, image_fname + '.png')
                img = load_img(img_path, grayscale=False,
                               target_size=self.target_size)
            img = img_to_array(img, data_format=self.data_format)

            # Load the subject.
            sub_path = os.path.join(self.data_dir, subject_fname)
            s_region = load_img(sub_path, grayscale=True,
                                target_size=self.target_size)
            s_region = img_to_array(s_region, data_format=self.data_format)

            # Load the object.
            obj_path = os.path.join(self.data_dir, object_fname)
            o_region = load_img(obj_path, grayscale=True,
                                target_size=self.target_size)
            o_region = img_to_array(o_region, data_format=self.data_format)

            # Create the batch.
            batch_image[i] = img
            batch_rel[i] = rel
            batch_s_regions[i] = s_region.flatten()
            batch_o_regions[i] = o_region.flatten()

        # Choose the inputs based on the parts of the relationship we will use.
        inputs = [batch_image]
        if self.use_subject:
            inputs.append(batch_rel[:, 0])
        if self.use_predicate:
            inputs.append(batch_rel[:, 1])
        if self.use_object:
            inputs.append(batch_rel[:, 2])
        return inputs, [batch_s_regions, batch_o_regions]
