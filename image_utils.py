import os
import cv2
import numpy as np


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def save_predictions(model, image_data, relationship_data, nb_iter, data_dir, input_dim):
    predictions = model.predict([image_data, relationship_data])
    predictions = predictions.reshape(input_dim, input_dim, 1)
    cv2.imwrite(os.path.join(data_dir, 'attention-' + str(nb_iter) + '.png'), predictions + image_data[0])
    #pickle.dump(predictions, open( os.path.join(data_dir, 'attention-' + str(nb_iter) + ".p"), "wb" ) )

