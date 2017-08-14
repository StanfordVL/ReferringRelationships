import os
import cv2
import numpy as np


def save_predictions(model, image_data, relationship_data, nb_iter, data_dir, input_dim):
    predictions = model.predict([image_data, relationship_data])
    predictions = predictions.reshape(input_dim, input_dim, 1)
    cv2.imwrite(os.path.join(data_dir, 'attention-' + str(nb_iter) + '.png'), predictions + image_data[0])
    #pickle.dump(predictions, open( os.path.join(data_dir, 'attention-' + str(nb_iter) + ".p"), "wb" ) )


def visualize_weights():
    saving attention heatmaps for one example
    subject_pred, object_pred = model.predict([train_images[k:k+1], train_subjects[k:k+1], train_predicates[k:k+1], train_objects[k:k+1]])
    subject_pred = subject_pred.reshape(input_dim, input_dim, 1)
    image_pred = np.zeros((input_dim, input_dim, 3))
    image_pred += subject_pred*255
    cv2.imwrite(os.path.join('results/2', 'attention-' + str(i) + '.png'), cv2.addWeighted(train_images[k], 0.6, image_pred, 0.4, 0))
    s_iou_mean, s_iou_acc, o_iou_mean, o_iou_acc = evaluate(model, train_images, train_subjects, train_predicates, train_objects, train_subject_bbox, train_object_bbox, iou_thresh, score_thresh)
