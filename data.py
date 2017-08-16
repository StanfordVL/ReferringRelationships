import json

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image


class VRDDataset():
    def __init__(self, num_subjects=100, num_predicates=70, num_objects=100, data_path="data/VRD/annotations_train.json", im_dim=224,
                 img_path='/data/chami/VRD/sg_dataset/sg_train_images/{}', im_metadata_path="data/VRD/image_metadata.json"):
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
        return (col_indexes * row_indexes).flatten()

    def get_train_val_splits(self, val_split):
        """
        :param val_split: float, proportion of validation examples
        :return: train image ids (list) and validation image ids (list)
        """
        image_idx = self.data.keys()[:10]
        for i, image_id in enumerate(image_idx):
            if i > len(image_idx)*(1.-val_split):
                self.val_image_idx += [image_id]
            else:
                self.train_image_idx += [image_id]
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
        return images, subject_regions, o_regions


class VisualGenomeRelationshipsDataset():
    def __init__(self, data_path="data/VisualGenome/relationships.json", im_dim=224,
                 im_metadata_path="data/VisualGenome/image_data.json", img_path='data/VisualGenome/images/{}.jpg'):
        self.data = json.load(open(data_path))
        self.im_metadata = json.load(open(im_metadata_path))
        self.relationships_to_idx = {}
        self.objects_to_idx = {}
        self.subjects_to_idx = {}
        self.objects_counter = 0
        self.relationships_counter = 0
        self.subjects_counter = 0
        self.nb_images = len(self.data)
        self.image_ids = []
        self.im_dim = im_dim
        self.relationships = []
        self.objects = []
        self.subjects = []
        self.objects_regions = []  # h, w, x, y
        self.subjects_regions = []  # h, w, x, y
        self.gt_regions = []
        self.col_template = np.arange(self.im_dim).reshape(1, self.im_dim)
        self.row_template = np.arange(self.im_dim).reshape(self.im_dim, 1)
        self.img_path = img_path

    def get_regions(self, obj, im_metadata):
        """
        :param object: object coordinates
        :param im_metadata: image size and width
        :return: rescaled top, left, bottom, right coordinates
        """
        h_ratio = self.im_dim * 1. / im_metadata['height']
        w_ratio = self.im_dim * 1. / im_metadata['width']
        height = int(obj["h"] * h_ratio)
        width = int(obj["w"] * w_ratio)
        x = int(obj["x"] * w_ratio)
        y = int(obj["y"] * h_ratio)
        return y, x, min(y + height, self.im_dim - 1), min(x + width, self.im_dim - 1)

    def get_object_idx(self, obj_name):
        if obj_name in self.objects_to_idx.keys():
            return self.objects_to_idx[obj_name]
        else:
            self.objects_to_idx[obj_name] = self.objects_counter
            self.objects_counter += 1
            return self.objects_to_idx[obj_name]

    def get_relationship_idx(self, rel_string):
        if rel_string in self.relationships_to_idx.keys():
            return self.relationships_to_idx[rel_string]
        else:
            self.relationships_to_idx[rel_string] = self.relationships_counter
            self.relationships_counter += 1
            return self.relationships_to_idx[rel_string]

    def get_subject_idx(self, subj_name):
        if subj_name in self.subjects_to_idx.keys():
            return self.subjects_to_idx[subj_name]
        else:
            self.subjects_to_idx[subj_name] = self.subjects_counter
            self.subjects_counter += 1
            return self.subjects_to_idx[subj_name]

    def get_object_name(self, rel_data, object_type):
        if "name" in rel_data[object_type].keys():
            return rel_data[object_type]["name"]
        else:
            return rel_data[object_type]["names"][0]

    def get_indexes(self, region, col_template, row_template):
        top, left, bottom, right = region
        col_indexes = (1 * (col_template > left) * (col_template < right)).repeat(self.im_dim, 0)
        row_indexes = (1 * (row_template > top) * (row_template < bottom)).repeat(self.im_dim, 1)
        return col_indexes * row_indexes

    def build_gt_regions(self, o_region, s_region):
        o_indexes = self.get_indexes(o_region, self.col_template, self.row_template)
        s_indexes = self.get_indexes(s_region, self.col_template, self.row_template)
        return 1 * ((o_indexes + s_indexes) > 0)

    def build_dataset(self):
        for i in range(self.nb_images):
            image_index = self.data[i]["image_index"]
            image_id = self.data[i]["image_id"]
            im_metadata = self.im_metadata[image_index]
            relationships = self.data[i]["relationships"]
            for j, rel_data in enumerate(relationships):
                object_name = self.get_object_name(rel_data, "object")
                # object_id = self.get_object_idx(object_name)
                subject_name = self.get_object_name(rel_data, "subject")
                # subject_id = self.get_object_idx(subject_name)
                rel_name = rel_data["predicate"]
                # relationship_id = self.get_relationship_idx(rel_data["predicate"])
                # self.relationships += [(object_id, relationship_id, subject_id)]
                relationship_id = self.get_relationship_idx(rel_name)
                object_id = self.get_object_idx(object_name)
                subject_id = self.get_subject_idx(subject_name)
                # self.relationships += [(object_id, relationship_id, subject_id)]
                o_region = self.get_regions(rel_data["object"], im_metadata)
                s_region = self.get_regions(rel_data["subject"], im_metadata)
                self.relationships += [relationship_id]
                self.objects += [object_id]
                self.subjects += [subject_id]
                self.objects_regions += [o_region]
                self.subjects_regions += [s_region]
                self.image_ids += [image_id]
                self.gt_regions += [self.build_gt_regions(o_region, s_region)]
        # self.objects_regions = np.array(self.objects_regions)
        # self.subjects_regions = np.array(self.subjects_regions)
        self.image_ids = np.array(self.image_ids)  # todo: these should not be class attributes
        self.relationships = np.array(self.relationships)
        self.objects = np.array(self.objects)
        self.subjects = np.array(self.subjects)
        self.gt_regions = np.array(self.gt_regions)
        return self.image_ids, self.subjects, self.relationships, self.objects, self.gt_regions

    def get_image_from_img_id(self, img_id):
        img = image.load_img(self.img_path.format(img_id), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array[0]

    def get_images(self, image_ids):
        images = np.zeros((len(image_ids), self.im_dim, self.im_dim, 3))
        for i, img_id in enumerate(image_ids):
            images[i] = self.get_image_from_img_id(img_id)
        return images


if __name__ == "__main__":
    data = VRDDataset()
    image_ids, subjects, relationships, objects, subject_regions, object_regions = data.build_dataset()
    images = data.get_images(image_ids)


    # ********************************** OLD CODE *********************************************

    # TODO: add split train-val-test
    # def get_object_idx(self, object_name):
    #     if object_name in self.objects_to_idx.keys():
    #         return self.objects_to_idx[object_name]
    #     else:
    #         self.objects_to_idx[object_name] = float(self.objects_counter)
    #         self.objects_counter += 1
    #         return self.objects_to_idx[object_name]

    # def _get_relationship_idx(self, predicate):
    #     if predicate in self.relationships_to_idx.keys():
    #         return self.relationships_to_idx[predicate]
    #     else:
    #         self.relationships_counter += 1
    #         self.relationships_to_idx[predicate] = self.relationships_counter
    #         return self.relationships_to_idx[predicate]
    # class VisualGenomeRelationshipsDataset(ReferringRelationshipsDataset):
    #     def __init__(self, data_path="data/VisualGenome/relationships.json", im_dim=224, im_metadata_path="data/VisualGenome/image_data.json", img_path='data/VisualGenome/images/{}.jpg'):
    #         super(self.__class__, self).__init__(data_path, im_dim, im_metadata_path, img_path)
