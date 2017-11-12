"""Util functions used for evaluating the model.
"""

from keras import backend as K

import tensorflow as tf


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    #y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
    #                   nb_classes)
    #y_true = K.reshape(y_true, (-1, nb_classes))
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(y_true[-1], tf.bool)
    y_true = tf.stack(unpacked[1:], axis=-1)

    return K.sum(tf.to_float(y_true * K.greater(K.argmax(y_true, axis=-1) - K.argmax(y_pred, axis=-1), 0))) / K.sum(tf.to_float(y_true))

def original_sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[1:], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))


def format_results(names, scalars):
    """Formats the results of training.

    Args:
        names: The names of the metrics.
        scalars: The values of the metrics.

    Returns:
        A string that contains the formatted scalars.
    """
    res = []
    for name, scalar in zip(names, scalars):
        res.append('%s: %2.3f' % (name, scalar))
    return ', '.join(res)

def pixel_acc(y_true, y_pred):
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    correct_preds =  K.mean(K.cast(K.equal(y_true, y_pred), "float32"), axis=1)
    return correct_preds

def mean_iu(y_true, y_pred):
    total = K.sum(y_true, axis=1)
    total_pred = K.sum(y_pred, axis=1)
    total_correct = K.sum(K.cast(K.equal(y_true, y_pred), "float32"), axis=1)
    iou = K.mean(total_correct / (K.epsilon() + total + total_pred - total_correct), axis=1)
    return iou

def iou(y_true, y_pred, heatmap_threshold):
    """Measures the mean IoU of our predictions with ground truth.

    Args:
        y_true: The ground truth bounding box locations.
        y_pred: Our heatmap predictions.
        heatmap_threshold: Config specified theshold above which we consider
          a prediction to contain an object.

    Returns:
        A float containing the mean IoU of our predictions.
    """
    pred = K.cast(K.greater(y_pred, heatmap_threshold), "float32")
    intersection = y_true * pred
    union = K.cast(K.greater(y_true + pred, 0), "float32")
    iou_values = K.sum(intersection, axis=1) / (K.epsilon() + K.sum(union, axis=1))
    return K.mean(iou_values)


def precision(y_true, y_pred, heatmap_threshold):
    """Measures the precision of our predictions with ground truth.

    Args:
        y_true: The ground truth bounding box locations.
        y_pred: Our heatmap predictions.
        heatmap_threshold: Config specified theshold above which we consider
          a prediction to contain an object.

    Returns:
        A float containing the precision of our predictions.
    """
    pred = K.cast(K.greater(y_pred, heatmap_threshold), "float32")
    tp = K.sum(y_true * pred, axis=1)
    fp = K.sum(K.cast(K.greater(pred - y_true, 0), 'float32'), axis=1)
    precision_values = tp/(tp + fp + K.epsilon())
    return K.mean(precision_values)


def recall(y_true, y_pred, heatmap_threshold):
    """Measures the recall of our predictions with ground truth.

    Args:
        y_true: The ground truth bounding box locations.
        y_pred: Our heatmap predictions.
        heatmap_threshold: Config specified theshold above which we consider
          a prediction to contain an object.

    Returns:
        A float containing the recall of our predictions.
    """
    pred = K.cast(K.greater(y_pred, heatmap_threshold), "float32")
    tp = K.sum(y_true * pred, axis=1)
    fn = K.sum(K.cast(K.greater(y_true - pred, 0), 'float32'), axis=1)
    recall_values = tp/(tp + fn + K.epsilon())
    return K.mean(recall_values)


def iou_acc(y_true, y_pred, heatmap_threshold):
    """Measures the mean accuracy of our predictions with ground truth.

    Here we consider an object localization to be correct if it contains an
    IoU > 0.5 with the ground truth box.

    Args:
        y_true: The ground truth bounding box locations.
        y_pred: Our heatmap predictions.
        heatmap_threshold: Config specified theshold above which we consider
          a prediction to contain an object.

    Returns:
        A float containing the mean accuracy of our predictions.
    """
    pred = K.cast(K.greater(y_pred, heatmap_threshold), "float32")
    intersection = y_true * pred
    union = K.cast(K.greater(y_true + pred, 0), "float32")
    iou_values = K.sum(intersection, axis=1) / (K.epsilon() + K.sum(union, axis=1))
    acc = K.cast(K.greater(iou_values, 0.5), "float32")
    return K.mean(acc)


def iou_bbox(y_true, y_pred, heatmap_threshold, input_dim):
    """Measures the mean IoU of our bbox predictions with ground truth.

    Args:
        y_true: The ground truth bounding box locations.
        y_pred: Our heatmap predictions.
        heatmap_threshold: Config specified theshold above which we consider
          a prediction to contain an object.

    Returns:
        A float containing the mean accuracy of our bbox predictions.
    """
    pred = K.cast(K.greater(y_pred, heatmap_threshold), "float32")
    pred = K.reshape(pred, (-1, input_dim, input_dim))
    horiz = K.sum(pred, axis=1, keepdims=True)
    horiz = K.cast(K.greater(horiz, 0), "float32")
    mask_horiz = K.repeat_elements(horiz, input_dim, axis=1)
    vert = K.sum(pred, axis=2, keepdims=True)
    vert = K.cast(K.greater(vert, 0), "float32")
    mask_vert = K.repeat_elements(vert, input_dim, axis=2)
    mask = mask_vert * mask_horiz
    mask = K.reshape(mask, (-1, input_dim * input_dim))
    intersection = y_true * mask
    union = K.cast(K.greater(y_true + mask, 0), "float32")
    iou_values = K.sum(intersection, axis=1) / (K.epsilon() + K.sum(union, axis=1))
    return K.mean(iou_values)


if __name__ == "__main__":
    import numpy as np;
    x = np.random.random((3, 3))
    input_dim = 3
    y = np.array([[0., 1., 1.],[1., 1., 1.],[1., 0, 1.]])
    print(x)
    print(y)
    tf.InteractiveSession()
    print(iou_bbox(y, x, 0.6, input_dim).eval())
