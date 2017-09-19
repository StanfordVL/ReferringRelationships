import os
import keras.backend as K
import numpy as np

from keras.preprocessing.image import Iterator, load_img, img_to_array


#
# def save_attention_weights(pdf_image, pred, save_path, img_height, img_width):
#     # save attention weights for one example
#     pred = pred.reshape(img_height, img_width, 1) * 255.
#     res = np.zeros((img_height, img_width, 3))
#     res += pred
#     attention_viz = cv2.addWeighted(pdf_image.astype(np.float), 0.4, res.astype(np.float), 0.6, 0)
#     cv2.imwrite(save_path, attention_viz)


class RefRelDataIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    """

    def __init__(self, image_dir, data_dir, input_dim, batch_size, shuffle=True, seed=None):
        data_format = K.image_data_format()
        self.data_dir = data_dir
        self.input_dim = input_dim
        self.target_size = (input_dim, input_dim)
        self.data_format = data_format
        self.image_dir = image_dir
        if self.data_format == 'channels_last':
            self.rgb_image_shape = self.target_size + (3,)
            self.gray_image_shape = self.target_size + (1,)
        else:
            self.image_shape = (3,) + self.target_size
            self.gray_image_shape = (1,) + self.target_size
        self.rel_idx = np.load(os.path.join(self.data_dir, "rel_idx.npy"))
        self.relationships = np.load(os.path.join(self.data_dir, "relationships.npy"))
        self.samples = self.rel_idx.shape[0]
        super(RefRelDataIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_image = np.zeros((current_batch_size,) + self.rgb_image_shape, dtype=K.floatx())
        batch_rel = np.zeros((current_batch_size,) + (3,), dtype=K.floatx())
        # batch_s_regions = np.zeros((current_batch_size,) + self.gray_image_shape, dtype=K.floatx())
        # batch_o_regions = np.zeros((current_batch_size,) + self.gray_image_shape, dtype=K.floatx())
        # todo: fix this shape thing
        batch_s_regions = np.zeros((current_batch_size,) + (self.input_dim * self.input_dim,), dtype=K.floatx())
        batch_o_regions = np.zeros((current_batch_size,) + (self.input_dim * self.input_dim,), dtype=K.floatx())
        # build batch of image data
        for i, j in enumerate(index_array):
            rel_id = self.rel_idx[j]
            rel = self.relationships[j]
            image_fname = "".join(rel_id.split("-")[:-1])
            subject_fname = rel_id + "-s.jpg"
            object_fname = rel_id + "-o.jpg"
            try:
                img = load_img(os.path.join(self.image_dir, image_fname + ".jpg"), grayscale=False,
                               target_size=self.target_size)
            except IOError:
                img = load_img(os.path.join(self.image_dir, image_fname + ".png"), grayscale=False,
                               target_size=self.target_size)
            s_region = load_img(os.path.join(self.data_dir, subject_fname), grayscale=True,
                                target_size=self.target_size)
            o_region = load_img(os.path.join(self.data_dir, object_fname), grayscale=True, target_size=self.target_size)
            img = img_to_array(img, data_format=self.data_format)
            s_region = img_to_array(s_region, data_format=self.data_format)  # .flatten()
            o_region = img_to_array(o_region, data_format=self.data_format)  # .flatten()
            batch_image[i] = img
            batch_rel[i] = rel
            batch_s_regions[i] = s_region.flatten()
            batch_o_regions[i] = o_region.flatten()
        return [batch_image, batch_rel[:, 0], batch_rel[:, 1], batch_rel[:, 2]], [batch_s_regions, batch_o_regions]
