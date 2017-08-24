import json
import os

import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image


class VRDDataset():
    def __init__(self, num_subjects=100, num_predicates=70, num_objects=100,
                 data_path="data/VRD/annotations_train.json", im_dim=224,
                 img_path='/data/chami/VRD/sg_dataset/sg_train_images/{}',
                 im_metadata_path="data/VRD/image_metadata.json"):
        self.data = json.load(open(data_path))
        self.im_metadata = json.load(open(im_metadata_path))
        self.im_dim = im_dim
        self.col_template = np.arange(self.im_dim).reshape(1, self.im_dim)
        self.row_template = np.arange(self.im_dim).reshape(self.im_dim, 1)
        self.img_path = img_path
        self.train_image_idx = []
        self.val_image_idx = []
        self.num_subjects = num_subjects
        self.num_predicates = num_predicates
        self.num_objects = num_objects

    def rescale_bbox_coordinates(self, obj, im_metadata):
        """
        :param object: object coordinates
        :param im_metadata: image size and width
        :return: rescaled top, left, bottom, right coordinates
        """
        h_ratio = self.im_dim * 1. / im_metadata['height']
        w_ratio = self.im_dim * 1. / im_metadata['width']
        y_min, y_max, x_min, x_max = obj["bbox"]
        return y_min * h_ratio, x_min * w_ratio, min(y_max * h_ratio, self.im_dim - 1), min(x_max * w_ratio,
                                                                                            self.im_dim - 1)

    def get_regions_from_bbox(self, bbox):
        """
        :param bbox: tuple (top, left, bottom, right) coordinates of the object of interest
        :return: converts bbox given as to image array with 0 or 1 for ground truth regions
        """
        top, left, bottom, right = bbox
        col_indexes = (1 * (self.col_template > left) * (self.col_template < right)).repeat(self.im_dim, 0)
        row_indexes = (1 * (self.row_template > top) * (self.row_template < bottom)).repeat(self.im_dim, 1)
        return col_indexes * row_indexes

    def get_train_val_splits(self, val_split, shuffle=True):
        """
        :param val_split: float, proportion of validation examples
        :return: train image ids (list) and validation image ids (list)
        """
        image_idx = self.data.keys()[:10]
        if shuffle:
            np.random.shuffle(image_idx)
        thresh = int(len(image_idx) * (1. - val_split))
        self.train_image_idx = image_idx[:thresh]
        self.val_image_idx = image_idx[thresh:]
        return self.train_image_idx, self.val_image_idx

    def build_dataset(self, image_idx):
        """
        :param image_idx: list of image ids
        :return: images ids (each image ids is repeated for each relationship within that image),
        relationships (Nx3 array with subject, predicate and object categories)
        subject and object bounding boxes (each Nx4)
        """
        subjects_bbox = []
        objects_bbox = []
        relationships = []
        image_ids = []
        for i, image_id in enumerate(image_idx):
            im_data = self.im_metadata[image_id]
            for j, relationship in enumerate(self.data[image_id]):
                image_ids += [image_id]
                subject_id = relationship["subject"]["category"]
                relationship_id = relationship["predicate"]
                object_id = relationship["object"]["category"]
                relationships += [(subject_id, relationship_id, object_id)]
                s_region = self.rescale_bbox_coordinates(relationship["subject"], im_data)
                o_region = self.rescale_bbox_coordinates(relationship["object"], im_data)
                subjects_bbox += [s_region]
                objects_bbox += [o_region]
        return np.array(image_ids), np.array(relationships), np.array(subjects_bbox), np.array(objects_bbox)

    def build_and_save_dataset(self, image_idx, save_dir):
        """
        :param image_idx: list of image ids
        :return: images ids (each image ids is repeated for each relationship within that image),
        relationships (Nx3 array with subject, predicate and object categories)
        subject and object bounding boxes (each Nx4)
        """
        rel_idx = []
        relationships = []
        for i, image_id in enumerate(image_idx):
            im_data = self.im_metadata[image_id]
            for j, relationship in enumerate(self.data[image_id]):
                rel_id = image_id.replace(".jpg", "") + "-{}".format(j)
                rel_idx += [rel_id]
                subject_id = relationship["subject"]["category"]
                predicate_id = relationship["predicate"]
                object_id = relationship["object"]["category"]
                relationships += [(subject_id, predicate_id, object_id)]
                s_bbox = self.rescale_bbox_coordinates(relationship["subject"], im_data)
                o_bbox= self.rescale_bbox_coordinates(relationship["object"], im_data)
                s_region = self.get_regions_from_bbox(s_bbox) * 255
                o_region = self.get_regions_from_bbox(o_bbox) * 255#TODO:this is just to visualize regions, needs to me removed afterwards 
                cv2.imwrite(os.path.join(save_dir, "{}-s.jpg".format(rel_id)), s_region)
                cv2.imwrite(os.path.join(save_dir, "{}-o.jpg".format(rel_id)), o_region)
        np.save(os.path.join(save_dir, "rel_idx.npy"), np.array(rel_idx))
        np.save(os.path.join(save_dir, "relationships.npy"), np.array(relationships))

    def get_image_from_img_id(self, img_id):
        img = image.load_img(self.img_path.format(img_id), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array[0]

    def get_images(self, image_idx):
        images = np.zeros((len(image_idx), self.im_dim, self.im_dim, 3))
        for i, image_id in enumerate(image_idx):
            images[i] = self.get_image_from_img_id(image_id)
        return images

    def get_images_and_regions(self, image_idx, subject_bbox, object_bbox):
        m = len(image_idx)
        images = np.zeros((m, self.im_dim, self.im_dim, 3))
        s_regions = np.zeros((m, self.im_dim * self.im_dim))
        o_regions = np.zeros((m, self.im_dim * self.im_dim))
        for i, image_id in enumerate(image_idx):
            s_bbox = subject_bbox[i]
            o_bbox = object_bbox[i]
            images[i] = self.get_image_from_img_id(image_id)
            s_regions[i] = self.get_regions_from_bbox(s_bbox)
            o_regions[i] = self.get_regions_from_bbox(o_bbox)
        return images, s_regions, o_regions


if __name__ == "__main__":
    train_val_split_ratio = 0.2
    vrd_dataset = VRDDataset()
    train_split, val_split = vrd_dataset.get_train_val_splits(train_val_split_ratio)
    vrd_dataset.build_and_save_dataset(train_split, "/data/chami/VRD/train")
    vrd_dataset.build_and_save_dataset(val_split, "/data/chami/VRD/val")
