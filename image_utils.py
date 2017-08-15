import os
import cv2
import numpy as np


def visualize_weights(orig_image, pred, input_dim, epoch, name, res_dir):
    # saving attention heatmaps for one example
    pred = pred.reshape(input_dim, input_dim, 1)
    image_pred = np.zeros((input_dim, input_dim, 3))
    image_pred += pred*255
    attention_viz = cv2.addWeighted(orig_image, 0.6, image_pred, 0.4, 0)
    cv2.imwrite(os.path.join(res_dir, 'attention-' + name + '-' + str(epoch) + '.png'), attention_viz)
