import os
import json
import cv2
import numpy as np
import argparse

from keras.models import load_model
from ReferringRelationships.iterator import RefRelDataIterator
from ReferringRelationships.utils.tmp import iou_acc_3, iou_3, iou_bbox_3, iou_bbox_5, iou_bbox_6, iou_7, iou_5, iou_acc_5

def get_att_map(orig_image, subj_pred, obj_pred, input_dim, relationship):
    # computes attention heatmap for one example
    subj_pred = subj_pred.reshape(input_dim, input_dim, 1)
    obj_pred = obj_pred.reshape(input_dim, input_dim, 1)
    subj_pred = 255. * subj_pred
    obj_pred = 255. * obj_pred
    image_pred_subj = np.zeros((input_dim, input_dim, 3), dtype='float32')
    image_pred_subj += subj_pred  # subject in white
    attention_subj = cv2.addWeighted(orig_image, 0.2, image_pred_subj, 0.8, 0)
    cv2.putText(attention_subj, relationship[0], (100, 20), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    image_pred_obj = np.zeros((input_dim, input_dim, 3), dtype='float32')
    image_pred_obj += obj_pred  # object in red
    attention_obj = cv2.addWeighted(orig_image, 0.2, image_pred_obj, 0.8, 0)
    cv2.putText(attention_obj, relationship[2], (100, 20), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(orig_image, "-".join(relationship), (20, 20), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    final = np.concatenate((orig_image, attention_subj, attention_obj), axis=1)
    cv2.putText(final, "-".join(relationship), (20, 20), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return final 

def get_dict(vocab_dir):
    predicates = json.load(open(os.path.join(vocab_dir, "predicates.json"), "r"))
    obj_subj = json.load(open(os.path.join(vocab_dir, "objects.json"), "r"))
    return predicates, obj_subj

def parse_args():
    """Initializes a parser and reads the command line parameters.
    Returns:
        An object containing all the parameters.
    """
    parser = argparse.ArgumentParser(description='Referring Relationships Visualizations.')

    # Session parameters.
    parser.add_argument('--model', type=str, help='model path', default='/data/chami/ReferringRelationships/models/09_24_2017/6/model28-0.55.h5')
    parser.add_argument('--num_examples', type=int, help='number of examples to show', default=20)
    parser.add_argument('--vocab_dir', type=str, help='where to load the id to category mapping for object subjects and predicate', default='/afs/cs.stanford.edu/u/chami/ReferringRelationships/data/VRD')
    parser.add_argument('--save_dir', type=str, help='where to save the attention heatmaps visualizations', default='att')
    args = parser.parse_args()
    return args


class objdict(dict):
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


if __name__ == "__main__": 
    args = parse_args()
    params = objdict(json.load(open(os.path.join(os.path.dirname(args.model), "args.json"), "r")))
    input_dim = params.input_dim
    params.batch_size = args.num_examples
    val_generator = RefRelDataIterator(params.val_data_dir, params)
    x_val, y_val = val_generator.next()
    x_val, y_val = val_generator.next()
    predicates, obj_subj = get_dict(args.vocab_dir)
    font = cv2.FONT_HERSHEY_SIMPLEX
    model = load_model(args.model, custom_objects={'iou_0.3': iou_3, "iou_5": iou_5, "iou_7":iou_7, "iou_acc_5":iou_acc_5, 'iou_acc_0.3': iou_acc_3, 'iou_bbox_0.3': iou_bbox_3, 'iou_bbox_5': iou_bbox_5, 'iou_bbox_6': iou_bbox_6})
    preds = model.predict(x_val)
    model_name = os.path.basename(args.model)
    for i in range(len(x_val[0])):
        img, subj_id, pred_id, obj_id = x_val[0][i], int(x_val[1][i][0]), int(x_val[2][i][0]), int(x_val[3][i][0])
        relationship = [obj_subj[subj_id], predicates[pred_id], obj_subj[obj_id]]
        subj_pred = preds[0][i]
        obj_pred = preds[1][i]
        attention_map = get_att_map(img, subj_pred, obj_pred, input_dim, relationship)
        cv2.imwrite(os.path.join(args.save_dir, 'att-{}-'.format(i) + model_name+ '.png'), attention_map)#cv2.cvtColor(attention_map, cv2.COLOR_BGR2RGB))    
