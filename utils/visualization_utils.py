"""Utility functions used for visualization.
"""

from PIL import Image
from PIL import ImageDraw

import os
import json
import numpy as np
import argparse


def add_attention(original_image, heatmap, input_dim):
    """Adds a heatmap visualization to the original image.

    Args:
        original_image: A PIL representation of the original image.
        heatmap: A numpy representation of where the object is predicted to be.
        input_dim: The dimensions of the predicted heatmaps.

    Returns:
        The attended heatmap over the image.
    """
    image = original_image.resize((input_dim, input_dim))
    image = np.array(image)
    heatmap = heatmap.reshape(input_dim, input_dim) * 255.
    attended_image = np.zeros((input_dim, input_dim, 3))
    for c in range(3):
        attended_image[:, :, c] = heatmap
    attention = 0.2*image + 0.8*attended_image
    attention = Image.fromarray(attention.astype('uint8'), 'RGB')
    return attention


def get_bbox_from_heatmap(heatmap, threshold=0.5, input_dim=224):
    """Grabs the bbox that the heatmap represents.

    Args:
        heatmap: A numpy representation of where the object is.
        threshold: The min value of the threshold.
        input_dim: The dimensions of the image.

    Returns:
        A tuple containing (ymin, ymax, xmin, xmax).
    """
    heatmap = heatmap.copy()
    heatmap = heatmap.reshape((input_dim, input_dim))
    heatmap[heatmap < threshold] = 0
    rows = heatmap.sum(axis = 1).nonzero()
    cols = heatmap.sum(axis = 0).nonzero()
    ymin = np.min(rows)
    ymax = np.max(rows)
    xmin = np.min(cols)
    xmax = np.max(cols)
    return (ymin, ymax, xmin, xmax)


def add_bbox_to_image(image, bbox, color='red', width=3):
    """Adds a bounding box to the image.

    Args:
        image: A PIL image.
        bbox: (ymin, ymax, xmin, xmax) box.
        color: Color to draw the box with.

    Returns:
        A PIL image with the bounding box drawn.
    """
    output = image.copy()
    ymin, ymax, xmin, xmax = bbox
    draw = ImageDraw.Draw(output)
    for i in range(width):
        draw.rectangle(((xmin+i, ymin+i), (xmax+i, ymax+i)), outline=color)
    return output


def add_bboxes(original_image, subject_heatmap, object_heatmap, input_dim,
               threshold=0.5, subject_color='blue', object_color='green',
               width=3):
    """Creates the visualizations for a predicted subject and object map.

    Args:
        original_image: A PIL representation of the original image.
        subject_heatmap: A numpy representation of where the subject is predicted
            to be.
        object_heatmap: A numpy representation of where the object is predicted
            to be.
        input_dim: The dimensions of the predicted heatmaps.
        threshold: The min value of the threshold.
        subject_color: The color of the subject bbox.
        object_color: The color of the object bbox.
        width: The width of the rectangle to be drawn.

    Returns:
        The attended subject and object heatmap over the image, concatenated with
        the original image.
    """
    image = original_image.resize((input_dim, input_dim))
    s_bbox = get_bbox_from_heatmap(subject_heatmap, input_dim=input_dim,
                                   threshold=threshold)
    subject_image = add_bbox_to_image(image, s_bbox, color=subject_color,
                                      width=width)
    o_bbox = get_bbox_from_heatmap(object_heatmap, input_dim=input_dim,
                                   threshold=threshold)
    object_image = add_bbox_to_image(image, o_bbox, color=object_color,
                                     width=width)
    together = np.concatenate((image, subject_image, object_image), axis=1)
    together = Image.fromarray(together.astype('uint8'), 'RGB')
    return together


def get_att_map(original_image, subject_heatmap, object_heatmap, input_dim,
                relationship):
    """Creates the visualizations for a predicted subject and object map.

    Args:
        original_image: A PIL representation of the original image.
        subject_heatmap: A numpy representation of where the subject is predicted
            to be.
        object_heatmap: A numpy representation of where the object is predicted
            to be.
        input_dim: The dimensions of the predicted heatmaps.
        relationship: A tuple containing the names of the subject, predicate
            and object.

    Returns:
        The attended subject and object heatmap over the image, concatenated with
        the original image.
    """
    image = original_image.resize((input_dim, input_dim))
    image = np.array(image)
    image = image[:,:,:-1]
    subject_heatmap = subject_heatmap.reshape(input_dim, input_dim) * 255.
    object_heatmap = object_heatmap.reshape(input_dim, input_dim) * 255.
    subject_image = np.zeros((input_dim, input_dim, 3))
    object_image = np.zeros((input_dim, input_dim, 3))
    for c in range(3):
        subject_image[:, :, c] = subject_heatmap
        object_image[:, :, c] = object_heatmap
    subject_attention = 0.2*image + 0.8*subject_image
    object_attention = 0.2*image + 0.8*object_image

    together = np.concatenate((image, subject_attention, object_attention),
                              axis=1)
    together = Image.fromarray(together.astype('uint8'), 'RGB')
    #txt = Image.new('RGBA', together.size, (255,255,255,0))
    #fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 24)
    #d = ImageDraw.Draw(txt)
    #d.text((10,input_dim-30), ' - '.join(relationship), font=fnt, fill=(255,0,255,255))
    #out = Image.alpha_composite(together.convert('RGBA'), txt)
    #return out
    return together


def get_dict(vocab_dir):
    """Returns the mapping from categories to names of objects and predicates.

    Args:
        vocab_dir: Directory location of where these files live. This funciton
            assumes that the directory contains a `objects.json` and
            `predicates.json` files of lists of names.

    Returns:
        A tuple of mapping from their relative category to names.
    """
    cat2pred = json.load(open(os.path.join(vocab_dir, "predicates.json"), "r"))
    cat2obj = json.load(open(os.path.join(vocab_dir, "objects.json"), "r"))
    return cat2pred, cat2obj


class objdict(dict):
    """Converts dicts to objects.
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def parse_args():
    """Initializes a parser and reads the command line parameters.

    Returns:
        An object containing all the parameters.
    """
    parser = argparse.ArgumentParser(
        description='Referring Relationships Visualizations.')

    # Session parameters.
    parser.add_argument('--model', type=str,
                        default='models/09_24_2017/6/model28-0.55.h5',
                        help='model path')
    parser.add_argument('--num_examples', type=int, default=20,
                        help='number of examples to show')
    parser.add_argument('--vocab_dir', type=str, default='data/VRD',
                        help='location of object and predicate names')
    parser.add_argument('--save_dir', type=str, default='att',
                        help='where to save the attention heatmaps visualizations')
    args = parser.parse_args()
    return args
