import tensorflow as tf
from keras import backend as K 

def iou(y_true, y_pred, thresh):
    pred = K.cast(K.greater(y_pred, thresh), "float32")
    intersection = y_true * pred
    union = K.cast(K.greater(y_true + pred, 0), "float32")
    iou_values = K.sum(intersection, axis=1) / (K.epsilon() + K.sum(union, axis=1))
    return K.mean(iou_values)


def iou_3(y_true, y_pred):
    return iou(y_true, y_pred, 0.3)


def iou_5(y_true, y_pred):
    return iou(y_true, y_pred, 0.5)


def iou_7(y_true, y_pred):
    return iou(y_true, y_pred, 0.7)


def iou_9(y_true, y_pred):
    return iou(y_true, y_pred, 0.9)

if __name__ == "__main__":
    import numpy as np;
    x = np.random.random((3, 2))
    y = np.array([[0., 1.],[1., 1.],[1., 0.]])
    print(x)
    print(y)
    tf.InteractiveSession()
    print(iou_5(y, y).eval())
