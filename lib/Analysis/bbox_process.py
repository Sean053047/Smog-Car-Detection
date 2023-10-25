import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ? 1. Import customized Transformer first. At this time, __main__ module knows that transformer.
# ? 2. Then, import predict_image to load module. Because it already knows the customized transformer, it won't raise a AttributeError to load module.
# from svm.Transformer import HogTransformer
# from svm.predict import predict_image as svm_predict
# from sklearn.svm import SVC

import numpy as np
import torch


def smoke_bbox_extend(bbox: np.ndarray, height: int, width: int, OT_EXTEND_RATIO:float):
    x1, y1, x2, y2 = bbox
    extend_x, extend_y = int(OT_EXTEND_RATIO * (x2 - x1) / 2), int(
        OT_EXTEND_RATIO * (y2 - y1)
    )
    x1 -= extend_x
    x2 += extend_x
    y2 += extend_y

    return calibrate_bbox(np.array([x1,y1,x2,y2]), height, width)
    
def calibrate_bbox(bbox:np.ndarray, height:int , width: int ):
    x1, y1, x2, y2 = bbox
    
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = width if x2 > width else x2
    y2 = height if y2 > height else y2

    return np.array([x1, y1, x2, y2])

def crop_image(image: np.ndarray, bbox: np.ndarray):
    height, width = image.shape[0], image.shape[1]
    bbox = calibrate_bbox(bbox, height=height, width=width)
    x1, y1, x2, y2 = tuple(map(int, bbox))
    return image[y1:y2, x1:x2 ,:]

def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

def box_area_ratio(box1, box2):
    """
    Return area ratio of intersection-over-box1 (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise area ratio values
        for every element in boxes1 and boxes2
    """
    area1 = box_area(box1.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    area_ratio = inter / area1[:, None]

    return area_ratio

def box_center_distance(box1, box2):
    """
    Return center distance between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise area ratio values
        for every element in boxes1 and boxes2
    """
    
    # centers of boxes
    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = ((x_p - x_g) ** 2 + (y_p - y_g) ** 2) **(1/2)
    return centers_distance_squared

def get_center_point(box, shift_y=0.5, shift_x=0.5):

    cent_x = box[:, None, 0]*(1-shift_x) + box[:, None, 2]*shift_x
    cent_y = box[:, None, 1]*(1-shift_y) + box[:, None, 3]*shift_y
    return torch.cat((cent_x, cent_y), dim=1) 
     

def bbox_diagnol_distance(box):
    
    diag = ((box[:, 0]-box[:,2])**2 + (box[:,1]- box[:,3]))**(1/2)
    return diag