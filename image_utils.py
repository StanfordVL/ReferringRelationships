import cv2
import os


def save_predictions(model, image_data, relationship_data, nb_iter, data_dir, input_dim):
    predictions = model.predict([image_data, relationship_data])
    predictions = predictions.reshape(input_dim, input_dim, 1)
    cv2.imwrite(os.path.join(data_dir, 'attention-' + str(nb_iter) + '.png'), predictions)

