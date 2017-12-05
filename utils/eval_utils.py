"""Util functions used for evaluating the model.
"""

from keras import backend as K

import tensorflow as tf


def get_metrics(output_dim, heatmap_threshold):
    """Returns all the metrics with the corresponding thresholds.

    Args:
        output_dim: The size of the predictions.
        heatmap_threshold: The heatmap thresholds we are evaluating with.

    Returns:
        A list of metric functions that take in the ground truth and the
        predictins to evaluate the model.
    """
    metrics = []
    for metric_func in [iou, recall, precision]:
        for thresh in heatmap_threshold:
            metric = (lambda f, t: lambda gt, pred: f(gt, pred, t))(
                metric_func, thresh)
            metric.__name__ = metric_func.__name__ + '_' + str(thresh)
            metrics.append(metric)
    metrics += [kl, cc, sim]
    return metrics

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

def format_results_eval(names, scalars):
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

    res.append(' & '.join(names))
    res.append(' & '.join(['%2.4f' % x for x in scalars]))
    return '\n '.join(res)

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


def iou_bbox(y_true, y_pred, heatmap_threshold, output_dim):
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
    pred = K.reshape(pred, (-1, output_dim, output_dim))
    horiz = K.sum(pred, axis=1, keepdims=True)
    horiz = K.cast(K.greater(horiz, 0), "float32")
    mask_horiz = K.repeat_elements(horiz, output_dim, axis=1)
    vert = K.sum(pred, axis=2, keepdims=True)
    vert = K.cast(K.greater(vert, 0), "float32")
    mask_vert = K.repeat_elements(vert, output_dim, axis=2)
    mask = mask_vert * mask_horiz
    mask = K.reshape(mask, (-1, output_dim * output_dim))
    intersection = y_true * mask
    union = K.cast(K.greater(y_true + mask, 0), "float32")
    iou_values = K.sum(intersection, axis=1) / (K.epsilon() + K.sum(union, axis=1))
    return K.mean(iou_values)


def cc(y_true, y_pred):
    """Measure the cross correlation of our predictions with ground truth.

    Args:
         y_true: The ground truth bounding box locations.
         y_pred: Our heatmap predictions.

    Returns:
         A float containing the cross correlation of our predictions.
    """
    sigma_true = K.std(y_true, axis=1, keepdims=True)
    mu_true = K.mean(y_true, axis=1, keepdims=True)
    sigma_pred = K.std(y_pred, axis=1, keepdims=True)
    mu_pred = K.mean(y_pred, axis=1, keepdims=True)
    mu_sub = (y_true - mu_true) * (y_pred - mu_pred)
    cov = K.mean(mu_sub, axis=1, keepdims=True)
    return K.mean(cov/((sigma_pred * sigma_true) + K.epsilon()))


def sim(y_true, y_pred):
    """Measures the similarity between our predictions with ground truth.

    Args:
         y_true: The ground truth bounding box locations.
         y_pred: Our heatmap predictions.

    Returns:
         A float containing the similarity of our predictions.
    """
    y_true = y_true / (K.epsilon() + K.sum(y_true, axis=1, keepdims=True))
    y_pred = y_pred / (K.epsilon() + K.sum(y_pred, axis=1, keepdims=True))
    indices_1 = K.cast(K.greater(y_pred, y_true), "float32")
    indices_2 = K.cast(K.greater(y_true, y_pred), "float32")
    mini = K.sum((y_true * indices_1) + (y_pred * indices_2), axis=1)
    return K.mean(mini)


def kl(y_true, y_pred):
     """Measure the KL divergence of our predictions with ground truth.

    Args:
         y_true: The ground truth bounding box locations.
         y_pred: Our heatmap predictions.

    Returns:
         A float containing the KL divergence of our predictions.
     """
     norm_true = y_true / (K.sum(y_true, axis=1, keepdims=True) + K.epsilon())
     norm_pred = y_pred / (K.sum(y_pred, axis=1, keepdims=True) + K.epsilon())
     x = K.log(K.epsilon() + (norm_true/(K.epsilon() + norm_pred)))
     return K.mean(K.sum((x * norm_true), axis=1))

if __name__ == "__main__":
    import numpy as np;
    x = np.random.random((3, 3))
    output_dim = 3
    y = np.array([[0., 1., 1.],[1., 1., 1.],[1., 0, 1.]])
    print(x)
    print(y)
    tf.InteractiveSession()
    print(iou_bbox(y, x, 0.6, output_dim).eval())
