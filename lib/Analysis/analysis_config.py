import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.utils.Base_config import BaseAttrs

project_pth = Path(__file__).resolve().parent.parent.parent

class License(BaseAttrs):
    attrs = ["model_pth", 
             "imgsz", 
             "conf_thresh", 
             "iou_thresh", 
             "area_ratio_thresh", 
             "augment"]
    model_pth: str
    imgsz: int
    conf_thresh : float
    iou_thresh : float
    area_ratio_thresh: float
    augment : int

class Smoke(BaseAttrs):
    attrs = ["model_pth",
             "OT_EXTEND_RATIO", 
             "SVM_WIDTH_MIN", 
             "SVM_HEIGHT_MIN", 
             "imgsz", 
             "conf_thresh", 
             "iou_thresh", 
             "augment"]
    model_pth: str
    OT_EXTEND_RATIO: float
    SVM_WIDTH_MIN: int
    SVM_HEIGHT_MIN: int
    imgsz: int
    conf_thresh: float
    iou_thresh: float
    augment: int

class Parameters(BaseAttrs):
    SETTING_FILE = str(project_pth / Path("cfg") /  Path("analysis_settings.json"))
    attrs = ["frame_interval", 
             "cut_frame_thresh", 
             "shift_frame_ratio", 
             "Smoke", 
             "License", ]
    frame_interval: int
    cut_frame_thresh : int
    shift_frame_ratio : float
    Smoke : Smoke
    License : License
    
class Common(BaseAttrs):
    SETTING_FILE = str(project_pth / Path("cfg") / Path("common_settings.json") )
    attrs = [ 
        "temp_pth",
        "track_results_pth"
    ]
    temp_pth : str 
    track_results_pth : str

Parameters.initiate(module_name=__name__)
Parameters.save_info2json()

Common.initiate(module_name= __name__)
Common.save_info2json()

