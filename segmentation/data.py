"""Converts a dataset into the format we expect for training.
"""

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

import abc
import argparse
import collections
import h5py
import json
import numpy as np
import sys
import os


class Dataset(object):
    """Implements helper functions for parsing dataset.
    """

    def __init__(self, save_dir, segmentation_dir, img_dir, image_ids_file,
                 im_dim=224):
        """Constructor for the VRD dataset object.

        Args:
            segmentation_dir: Directory with all the segmentation masks.
            img_dir: Location of the images.
            im_dim: The size of images.
            image_ids_file: The filenames of the images.
        """
        self.save_dir = save_dir
        self.segmentation_dir = segmentation_dir
        self.im_dim = im_dim
        self.img_dir = img_dir
        self.image_ids = self.parse_image_ids_file(image_ids_file)

    def parse_image_ids_file(self, image_ids_file):
        """Parses the image ids in a text file.

        Args:
            image_ids_file: Location of the file.

        Returns:
            The list of image ids.
        """
        image_ids = []
        for line in open(image_ids_file):
            image_ids.append(line.strip())
        return sorted(image_ids)

    def get_image_from_img_id(self, img_id):
        """Reads the image associated with a specific img_id.

        Args:
            img_id: The if of the image to be read.

        Returns:
            The image as a numpy array.
        """
        try:
            img_path = os.path.join(self.img_dir, img_id +'.jpg')
        except:
            img_path = os.path.join(self.img_dir, img_id +'.png')
        img = image.load_img(img_path, target_size=(self.im_dim, self.im_dim))
        img_array = image.img_to_array(img)

        # Preprocess the image according to the network we are using.
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array[0]

    def save_images(self):
        """Preprocesses and saves the images.
        """
        # Grab the image ids.
        num_images = len(self.image_ids)

        # Create the image dataset.
        dataset = h5py.File(os.path.join(self.save_dir, 'images.hdf5'), 'w')
        images_db = dataset.create_dataset('images',
                                           (num_images,
                                            self.im_dim, self.im_dim, 3),
                                           dtype='f')

        # Iterate and save all the images first.
        for image_index, image_id in enumerate(self.image_ids):
            try:
                image = self.get_image_from_img_id(image_id)
            except KeyError:
                print('Image %s not found' % str(image_id))
                continue
            images_db[image_index] = image

            # Log the progress.
            if image_index % 100 == 0:
                print('| {}/{} images saved'.format(image_index, num_images))

    def parse_image(self, image):
        """Converts the image into the segmentation masks per class.
        """
        pixel_map = json.load(open('../data/pascal/pixel_map.json'))
        pixels = {}
        num_classes = 0
        for k in pixel_map.keys():
            pixel_list = json.loads(k)
            if pixel_map[k] not in pixels:
                pixels[pixel_map[k]] = []
                num_classes += 1
            pixels[pixel_map[k]].append(pixel_list)
        output = np.zeros((self.im_dim, self.im_dim, num_classes))
        for cls_index in range(num_classes):
            union = np.ones((self.im_dim, self.im_dim), dtype=np.int32)
            for pixel in pixels[cls_index]:
                intersection = np.ones((self.im_dim, self.im_dim), dtype=np.int32)
                for channel in range(len(pixel)):
                    intersection *= image[:, :, channel] == pixel[channel]
                union += intersection
            output[:, :, cls_index] = np.array(union > 0, dtype=np.int32)
        return output

    def get_segmentation_from_id(self, image_id):
        """Grabs the segmentation image and parses it.

        Args:
            image_id: The name of the segmentation file.

        Returns:
            A numpy array of size input_dim, input_dim, num_classes.
        """
        img_path = os.path.join(self.segmentation_dir, image_id +'.png')
        img = image.load_img(img_path, target_size=(self.im_dim, self.im_dim))
        img_array = image.img_to_array(img)
        return self.parse_image(img_array)

    def build_and_save_dataset(self):
        """Converts the dataset into format we will use for training.
        """
        # Grab the image ids.
        num_images = len(self.image_ids)

        # Create the dataset.
        dataset = h5py.File(os.path.join(self.save_dir, 'dataset.hdf5'), 'w')
        object_db = dataset.create_dataset('object_locations',
                                           (num_images, self.im_dim,
                                            self.im_dim, 21),
                                           dtype='f')

        # Iterate and count the number of relationships.
        for image_index, image_id in enumerate(self.image_ids):
            segmentations = self.get_segmentation_from_id(image_id)
            try:
                object_db[image_index] = segmentations
            except:
                print('Segmentation %s not found' % str(image_id))
                continue

            # Log the progress.
            if image_index % 100 == 0:
                print('| {}/{} images processed'.format(image_index, num_images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset creation for Pascal.')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='where to save the ground truth masks, this '
                        'Location where dataset should be saved.')
    parser.add_argument('--img-dir', type=str, default=None,
                        help='Location where images are stored.')
    parser.add_argument('--image-ids-file', type=str, default=None,
                        help='File containing the images in the set.')
    parser.add_argument('--segmentation-dir', type=str, default=None,
                        help='Json with relationships for each image.')
    parser.add_argument('--image-dim', type=int, default=224,
                        help='The size the images should be saved as.')
    parser.add_argument('--save-images', action='store_true',
                        help='Use this flag to specify that the images '
                        'should also be saved.')
    args = parser.parse_args()

    # Make sure that the required fields are present.
    if args.save_dir is None:
        print('--save-dir not specified. Exiting!')
        sys.exit(0)
    if args.img_dir is None:
        print('--img-dir not specified. Exiting!')
        sys.exit(0)
    if args.segmentation_dir is None:
        print('--segmentation-dir not specified. Exiting!')
        sys.exit(0)
    if args.image_ids_file is None:
        print('--image-ids-file not specified. Exiting!')
        sys.exit(0)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = Dataset(args.save_dir,
                      args.segmentation_dir,
                      args.img_dir,
                      args.image_ids_file,
                      im_dim=args.image_dim)
    if args.save_images:
        dataset.save_images()
    dataset.build_and_save_dataset()
