import tensorflow as tf
from keras import backend as K 

def iou_5(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.cast(y_true * y_pred > 0, tf.float32)
    union = tf.cast(y_true + y_pred > 0, tf.float32)
    iou_values = K.sum(intersection, axis=-1) / K.sum(union, axis=-1)
    return K.mean(iou_values)

def iou_7(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.7, tf.float32)
    intersection = tf.cast(y_true * y_pred > 0, tf.float32)
    union = tf.cast(y_true + y_pred > 0, tf.float32)
    iou_values = K.sum(intersection, axis=-1) / K.sum(union, axis=-1)
    return K.mean(iou_values)

def iou_9(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.9, tf.float32)
    intersection = tf.cast(y_true * y_pred > 0, tf.float32)
    union = tf.cast(y_true + y_pred > 0, tf.float32)
    iou_values = K.sum(intersection, axis=-1) / K.sum(union, axis=-1)
    return K.mean(iou_values)
