"""Converts a dataset into the format we expect for training.
"""

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

import abc
import argparse
import cv2
import h5py
import json
import numpy as np
import sys
import os


class Dataset(object):
    """Implements helper functions for parsing dataset.
    """

    def __init__(self, data_path, img_dir, im_metadata_path, im_dim=224):
        """Constructor for the VRD dataset object.

        Args:
            data_path: Annotations in the VRD dataset.
            img_dir: Location of the images.
            im_metadata_path: Location of the file containing image metadata.
            im_dim: The size of images.
        """
        self.data = json.load(open(data_path))
        self.im_metadata = json.load(open(im_metadata_path))
        self.im_dim = im_dim
        self.col_template = np.arange(self.im_dim).reshape(1, self.im_dim)
        self.row_template = np.arange(self.im_dim).reshape(self.im_dim, 1)
        self.img_dir = img_dir

    def rescale_bbox_coordinates(self, bbox, height, width):
        """Rescales the bbox coords according to the `im_dim`.

        Args:
            bbox: A tuple of (top, left, bottom, right) coordinates of the
                object of interest.
            height: original image height.
            width: original image width.
        Returns:
            A tuple containing the rescaled bbox coordinates.
        """
        h_ratio = self.im_dim * 1. / height
        w_ratio = self.im_dim * 1. / width
        y_min, y_max, x_min, x_max = bbox
        y0 = max(y_min * h_ratio, 0)
        x0 = max(x_min * w_ratio, 0)
        y1 = min(y_max * h_ratio, self.im_dim - 1)
        x1 = min(x_max * w_ratio, self.im_dim - 1)
        return y0, x0, y1, x1

    def get_regions_from_bbox(self, bbox):
        """Converts a bbox into a binary image for gt regions.

        Args:
            bbox: A tuple of (top, left, bottom, right) coordinates of the
              object of interest.

        Returns:
            An image array with 0 or 1 for ground truth regions.
        """
        top, left, bottom, right = bbox
        col_indexes = (1 * (self.col_template > left) *
                       (self.col_template < right)).repeat(self.im_dim, 0)
        row_indexes = (1 * (self.row_template > top) *
                       (self.row_template < bottom)).repeat(self.im_dim, 1)
        return (col_indexes * row_indexes).reshape((self.im_dim, self.im_dim))

    def get_train_val_splits(self, val_percent):
        """Splits the dataset into train and val splits.

        Args:
            val_percent: float, proportion of examples that should be in val.

        Returns:
            A tuple containing the image ids in the train and val sets.
        """
        image_ids = list(sorted(self.data.keys()))
        thresh = int(len(image_ids) * (1. - val_percent))
        train_image_ids = image_ids[:thresh]
        val_image_ids = image_ids[thresh:]
        return train_image_ids, val_image_ids

    def get_image_from_img_id(self, img_id):
        """Reads the image associated with a specific img_id.

        Args:
            img_id: The if of the image to be read.

        Returns:
            The image as a numpy array.
        """
        img_path = os.path.join(self.img_dir, img_id)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)

        # Preprocess the image according to the network we are using.
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array[0]

    def get_images(self, image_ids):
        """Loads all the images in the list of image_ids.

        Args:
            image_ids: A list of image ids.

        Returns:
            images: A list of numpy representations of the images.
        """
        images = np.zeros((len(image_ids), self.im_dim, self.im_dim, 3))
        for i, image_id in enumerate(image_ids):
            images[i] = self.get_image_from_img_id(image_id)
        return images

    def get_images_and_regions(self, image_ids, subject_bboxs, object_bboxs):
        """Grabs the image and subject-object locations.

        Args:
            image_ids: A list of image ids to load.
            subject_bboxes: A list of subject bboxes.
            object_bboxes: A list of object bboxes.

        Returns:
            A tuple containing a numpy representation of all the images,
            a numpy representation of all the subject locations,
            a numpry representation of all the object locations.
        """
        num_images = len(image_ids)
        images = np.zeros((num_images, self.im_dim, self.im_dim, 3))
        s_regions = np.zeros((num_images, self.im_dim * self.im_dim))
        o_regions = np.zeros((num_images, self.im_dim * self.im_dim))
        for i, image_id in enumerate(image_ids):
            s_bbox = subject_bboxes[i]
            o_bbox = object_bboxes[i]
            images[i] = self.get_image_from_img_id(image_id)
            s_regions[i] = self.get_regions_from_bbox(s_bbox)
            o_regions[i] = self.get_regions_from_bbox(o_bbox)
        return images, s_regions, o_regions

    @abc.abstractmethod
    def build_and_save_dataset(self, save_dir, image_ids=None):
        """Converts the dataset into format we will use for training.

        Converts the dataset into a series of images, relationship labels
        and heatmaps.

        Args:
            save_dir: Location to save the data.
            image_ids: List of image ids.
        """
        raise NotImplementedError


class SmartDataset(Dataset):
    """Parses the dataset into a format used for training.
    """

    def build_and_save_dataset(self, save_dir, image_ids=None):
        """Converts the dataset into format we will use for training.

        Converts the dataset into a series of images, relationship labels
        and heatmaps.

        Args:
            save_dir: Location to save the data.
            image_ids: List of image ids.
        """
        total_relationships = 0

        # Grab the image ids.
        if not image_ids:
            image_ids = sorted(self.data.keys())
        num_images = len(image_ids)

        # Create the image dataset.
        dataset = h5py.File(os.path.join(save_dir, 'dataset.hdf5'), 'w')
        images_db = dataset.create_dataset('images',
                                           (num_images,
                                            self.im_dim, self.im_dim, 3),
                                           dtype='f')

        # Iterate and save all the images first.
        for image_index, image_id in enumerate(image_ids):
            try:
                im_data = self.im_metadata[image_id]
                image = self.get_image_from_img_id(image_id)
            except KeyError:
                print('Image %s not found' % str(image_id))
                continue
            images_db[image_index] = image
            total_relationships += len(self.data[image_id])

            # Log the progress.
            if image_index % 100 == 0:
                print('| {}/{} images saved'.format(image_index, num_images))

        # Build the category and heatmap datasets.
        categories_db = dataset.create_dataset('categories',
                                               (total_relationships, 4),
                                               dtype='f')
        subject_db = dataset.create_dataset('subject_locations',
                                            (total_relationships,
                                             self.im_dim, self.im_dim),
                                            dtype='f')
        object_db = dataset.create_dataset('object_locations',
                                           (total_relationships,
                                            self.im_dim, self.im_dim),
                                           dtype='f')

        # Now save all the relationships.
        for image_index, image_id in enumerate(image_ids):
            seen = {}

            # Iterate over all the relationships in the image
            for j, relationship in enumerate(self.data[image_id]):
                subject_cat = relationship['subject']['category']
                predicate_cat = relationship['predicate']
                object_cat = relationship['object']['category']
                s_bbox = self.rescale_bbox_coordinates(
                    relationship['subject']['bbox'],
                    im_data['height'],
                    im_data['width'])
                o_bbox= self.rescale_bbox_coordinates(
                    relationship['object']['bbox'],
                    im_data['height'],
                    im_data['width'])
                s_region = self.get_regions_from_bbox(s_bbox)
                o_region = self.get_regions_from_bbox(o_bbox)
                seen_key = '_'.join([str(x) for x in
                                     [subject_cat, predicate_cat, object_cat]])
                if seen_key not in seen:
                    rel = {'image_index': image_index,
                           'subject': s_region,
                           'object': o_region,
                           'subject_cat': subject_cat,
                           'predicate_cat': predicate_cat,
                           'object_cat': object_cat}
                    seen[seen_key] = rel
                else:
                    rel = seen[seen_key]
                    rel['subject'] = (rel['subject'] + s_region
                                      - np.multiply(rel['subject'], s_region))
                    rel['object'] = (rel['object'] + o_region
                                     - np.multiply(rel['object'], o_region))

            for rel in seen.values():
                subject_db[i] = rel['subject']
                object_db[i] = rel['object']
                categories_db[i, 0] = rel['subject_cat']
                categories_db[i, 1] = rel['predicate_cat']
                categories_db[i, 2] = rel['object_cat']
                categories_db[i, 3] = rel['image_index']

            # Log the progress.
            if image_index % 100 == 0:
                print('| {}/{} images processed'.format(image_index, num_images))

        # Log the number of relationships we have seen.
        print("Total relationships in dataset: %d" % total_relationships)
        print("Total UNIQUE relationships per image: %d" % len(relationships))

        return relationships


class DuplicatedDataset(Dataset):
    """Parses the dataset into a format used for training.
    """

    def build_and_save_dataset(self, save_dir, image_ids=None):
        """Converts the dataset into format we will use for training.

        Converts the dataset into a series of images, relationship labels
        and heatmaps.

        Args:
            save_dir: Location to save the data.
            image_ids: List of image ids.
        """
        relationships = self.build_dataset(image_ids=image_ids)
        self.save_dataset(save_dir, relationships)

    def build_dataset(self, image_ids=None):
        """Converts the dataset into a list of relationships and categories.

        Args:
            image_ids: List of image ids.

        Returns:
            relationships: A list containing objects that contain:
                - 'image': The numpy image representation.
                - 'subject': The numpy subject location mask.
                - 'object': The numpy object location mask.
                - 'subject_cat': Int index of the subject category.
                - 'predicate_cat': Int index of the predicate category.
                - 'object_cat': Int index of the object category.
        """
        # Initialize output and logging variables.
        relationships = []
        total_relationships = 0

        # Grab the image ids.
        if not image_ids:
            image_ids = sorted(self.data.keys())
        num_images = len(image_ids)

        # Iterate over all the images
        for i, image_id in enumerate(image_ids):
            seen = {}
            try:
                im_data = self.im_metadata[image_id]
                image = self.get_image_from_img_id(image_id)
            except KeyError:
                print('Image %s not found' % str(image_id))
                continue
            total_relationships += len(self.data[image_id])

            # Iterate over all the relationships in the image
            for j, relationship in enumerate(self.data[image_id]):
                subject_cat = relationship['subject']['category']
                predicate_cat = relationship['predicate']
                object_cat = relationship['object']['category']
                s_bbox = self.rescale_bbox_coordinates(
                    relationship['subject']['bbox'],
                    im_data['height'],
                    im_data['width'])
                o_bbox= self.rescale_bbox_coordinates(
                    relationship['object']['bbox'],
                    im_data['height'],
                    im_data['width'])
                s_region = self.get_regions_from_bbox(s_bbox)
                o_region = self.get_regions_from_bbox(o_bbox)
                seen_key = '_'.join([str(x) for x in
                                     [subject_cat, predicate_cat, object_cat]])
                if seen_key not in seen:
                    rel = {'image': image,
                           'subject': s_region,
                           'object': o_region,
                           'subject_cat': subject_cat,
                           'predicate_cat': predicate_cat,
                           'object_cat': object_cat}
                    seen[seen_key] = rel
                else:
                    rel = seen[seen_key]
                    rel['subject'] = (rel['subject'] + s_region
                                      - np.multiply(rel['subject'], s_region))
                    rel['object'] = (rel['object'] + o_region
                                     - np.multiply(rel['object'], o_region))

            relationships.extend(seen.values())

            # Log the progress.
            if i % 100 == 0:
                print('| {}/{} images processed'.format(i, num_images))

        # Log the number of relationships we have seen.
        print("Total relationships in dataset: %d" % total_relationships)
        print("Total UNIQUE relationships per image: %d" % len(relationships))

        return relationships

    def save_dataset(self, save_dir, relationships):
        """Converts the dataset into hdf5.

        Args:
            save_dir: Location to save the data.
            relationships: A list containing objects that contain:
                - 'image': The numpy image representation.
                - 'subject': The numpy subject location mask.
                - 'object': The numpy object location mask.
                - 'subject_cat': Int index of the subject category.
                - 'predicate_cat': Int index of the predicate category.
                - 'object_cat': Int index of the object category.
        """
        total_relationships = len(relationships)
        dataset = h5py.File(os.path.join(save_dir, 'dataset.hdf5'), 'w')
        images_db = dataset.create_dataset('images',
                                           (total_relationships,
                                            self.im_dim, self.im_dim, 3),
                                           dtype='f')
        categories_db = dataset.create_dataset('categories',
                                               (total_relationships, 3),
                                               dtype='f')
        subject_db = dataset.create_dataset('subject_locations',
                                            (total_relationships,
                                             self.im_dim, self.im_dim),
                                            dtype='f')
        object_db = dataset.create_dataset('object_locations',
                                           (total_relationships,
                                            self.im_dim, self.im_dim),
                                           dtype='f')
        for i, rel in enumerate(relationships):
            images_db[i] = rel['image']
            subject_db[i] = rel['subject']
            object_db[i] = rel['object']
            categories_db[i, 0] = rel['subject_cat']
            categories_db[i, 1] = rel['predicate_cat']
            categories_db[i, 2] = rel['object_cat']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset creation for Visual '
                                     'Relationship model. This scripts saves '
                                     'masks for objects and subjects in '
                                     'directories, as well as numpy arrays '
                                     'for relationships.')
    parser.add_argument('--test', action='store_true',
                        help='When true, the data is not split into training '
                        'and validation sets')
    parser.add_argument('--val-percent', type=float, default=0.1,
                        help='Fraction of images in validation split.')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='where to save the ground truth masks, this '
                        'Location where dataset should be saved.')
    parser.add_argument('--img-dir', type=str, default=None,
                        help='Location where images are stored.')
    parser.add_argument('--annotations', type=str,
                        default='data/VRD/annotations_train.json',
                        help='Json with relationships for each image.')
    parser.add_argument('--image-metadata', type=str,
                        default='data/VRD/train_image_metadata.json',
                        help='Image metadata json file.')
    parser.add_argument('--image-dim', type=int, default=224,
                        help='The size the images should be saved as.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='The random seed used to reproduce results.')
    args = parser.parse_args()

    # Make sure that the required fields are present.
    if args.save_dir is None:
        print('--save-dir not specified. Exiting!')
        sys.exit(0)
    if args.img_dir is None:
        print('--img-dir not specified. Exiting!')
        sys.exit(0)
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    # set the random seed.
    np.random.seed(args.seed)

    dataset = SmartDataset(args.annotations, args.img_dir,
                         args.image_metadata, im_dim=args.image_dim)
    if args.test:
        # Build the test dataset.
        test_dir = os.path.join(args.save_dir, 'test')
        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)
        dataset.build_and_save_dataset(test_dir)
    else:
        # Split the images into train and val datasets.
        train_split, val_split = dataset.get_train_val_splits(
            args.val_percent)

        # Build the validation dataset.
        val_dir = os.path.join(args.save_dir, 'val')
        if not os.path.isdir(val_dir):
            os.mkdir(val_dir)
        print('| Building validation data...')
        dataset.build_and_save_dataset(val_dir, image_ids=val_split)

        # Build the training dataset.
        train_dir = os.path.join(args.save_dir, 'train')
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)
        print('| Building training data...')
        dataset.build_and_save_dataset(train_dir, image_ids=train_split)
