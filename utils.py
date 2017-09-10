import os
import cv2
import numpy as np


def visualize_weights(orig_image, pred, input_dim, epoch, name, res_dir):
    # saving attention heatmaps for one example
    pred = pred.reshape(input_dim, input_dim, 1)
    pred = 1.*(pred>0.9)
    image_pred = np.zeros((input_dim, input_dim, 3))
    image_pred += pred*255
    attention_viz = cv2.addWeighted(orig_image, 0.6, image_pred, 0.4, 0)
    cv2.imwrite(os.path.join(res_dir, 'attention-' + name + '-' + str(epoch) + '.png'), attention_viz)


def format_params(params):
    stars = "*" * 30
    formatted_params = ""
    for params_type in params.keys():
        formatted_params += stars + params_type + stars + "\n\n"
        for x in params[params_type].keys():
            formatted_params += "{} : {} \n".format(x, params[params_type][x])
        formatted_params += "\n"
    return formatted_params


def get_dir_name(models_dir):
    existing_dirs = np.array([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]).astype(
        np.int)
    if len(existing_dirs) > 0:
        return os.path.join(models_dir, str(existing_dirs.max() + 1))
    else:
        return os.path.join(models_dir, '1')