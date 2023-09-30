import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent)
from analysis_config import Parameters as analysis_par
import numpy as np

print(analysis_par.Smoke)
OT_EXTEND_RATIO = analysis_par.Smoke.OT_EXTEND_RATIO

def smoke_bbox_extend(bbox: np.ndarray, height: int, width: int):
    
    x1, y1, x2, y2 = bbox
    extend_x, extend_y = int(OT_EXTEND_RATIO * (x2 - x1) / 2), int(
        OT_EXTEND_RATIO * (y2 - y1)
    )
    x1 -= extend_x
    x2 += extend_x
    y2 += extend_y

    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = width if x2 > width else x2
    y2 = height if y2 > height else y2

    return np.array([x1, y1, x2, y2])


def crop_image(image: np.ndarray, bbox: np.ndarray):
    x1, y1, x2, y2 = tuple(map(int, bbox))
    return image[y1:y2, x1:x2 ,:]