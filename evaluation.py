import numpy as np

def get_bbox_from_heatmap(heatmap, score_thresh):
    heatmap = 1 * (heatmap > score_thresh)
    x_non_zero = heatmap.sum(axis=0).nonzero()[0]
    y_non_zero = heatmap.sum(axis=1).nonzero()[0]
    if len(x_non_zero) > 0 and len(y_non_zero) > 0:
        x_min, x_max = x_non_zero[0], x_non_zero[-1]
        y_min, y_max = y_non_zero[0], y_non_zero[-1]
        return y_min, x_min, y_max, x_max
    else:
        return None


def do_overlap(bbox1, bbox2):
    """
    :param bbox1: bounding box of the first rectangle
    :param bbox2: bounding box of the second rectangle
    :return: 1 if the two rectangles overlap
    """
    if bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0]:
        return False
    if bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]:
        return False
    return True


def compute_iou(bbox1, bbox2):
    """
    :param bbox1: (top, left, bottom, right)
    :param bbox2: (top, left, bottom, right)
    :return: intersection over union
    """
    try:
        top_1, left_1, bottom_1, right_1 = bbox1
        top_2, left_2, bottom_2, right_2 = bbox2
        if do_overlap((top_1, left_1, bottom_1, right_1), (top_2, left_2, bottom_2, right_2)):
            intersection = (min(bottom_1, bottom_2) - max(top_1, top_2)) * (min(right_1, right_2) - max(left_1, left_2))
            union = (bottom_1 - top_1) * (right_1 - left_1) + (bottom_2 - top_2) * (right_2 - left_2) - intersection
            return float(intersection) / float(union)
        return 0.
    except:
        return 0


def evaluate(s_regions_pred, o_regions_pred, val_subject_bbox, val_object_bbox, input_dim, score_thresh):
    s_iou = []
    o_iou = []
    for i in range(len(s_regions_pred)):
        pred_subject_bbox = get_bbox_from_heatmap(s_regions_pred[i].reshape(input_dim, input_dim), score_thresh)
        pred_object_bbox = get_bbox_from_heatmap(o_regions_pred[i].reshape(input_dim, input_dim), score_thresh)
        s_iou += [compute_iou(pred_subject_bbox, val_subject_bbox[i])]
        o_iou += [compute_iou(pred_object_bbox, val_object_bbox[i])]
    s_iou = np.array(s_iou)
    o_iou = np.array(o_iou)
    return s_iou, o_iou
