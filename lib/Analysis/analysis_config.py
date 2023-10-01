import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.utils.Base_config import BaseAttrs

class License(BaseAttrs):
    attrs = ["model_pth", "imgsz", "license_conf_thresh", "iou_thresh", "area_ratio_thresh"]
    model_pth: str
    imgsz: int
    license_conf_thresh : float
    iou_thresh : float
    area_ratio_thresh: float
    
class Smoke(BaseAttrs):
    attrs = ["model_pth","OT_EXTEND_RATIO", "SVM_WIDTH_MIN", "SVM_HEIGHT_MIN"]
    model_pth: str
    OT_EXTEND_RATIO: float
    SVM_WIDTH_MIN: int
    SVM_HEIGHT_MIN: int
    
class Parameters(BaseAttrs):
    SETTING_FILE = str(Path(__file__).resolve().parent / Path("cfg") /  Path("analysis_settings.json"))
    attrs = ["FRAME_INTERVAL", "Smoke", "License"]
    FRAME_INTERVAL: int
    Smoke : Smoke
    License : License
        
Parameters.initiate(module_name=__name__)
Parameters.save_info2json()

