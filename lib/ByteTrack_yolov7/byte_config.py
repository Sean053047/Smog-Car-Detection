import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.utils.Base_config import BaseAttrs
import json

project_pth = Path(__file__).resolve().parent.parent.parent

class Parameters(BaseAttrs):
    SETTING_FILE = str(
        project_pth / Path("cfg") / Path("ByteTrack_settings.json")
    )
    attrs = [
        "model_pth",
        "conf_thresh",
        "iou_thresh",
        "track_thresh",
        "track_buffer",
        "match_thresh",
        "min_box_area",
        "imgsz",
        "augment"
    ]
    model_pth : str
    conf_thresh : float 
    iou_thresh : float
    track_thresh : float
    track_buffer : int
    match_thresh : float 
    min_box_area : int 
    imgsz : int
    augment : int

    OT_TYPE_FILE = Path(__file__).resolve().parent / Path("model") / Path("OT-type.json")
    @classmethod
    def load_OT_type(cls):
        if cls.OT_TYPE_FILE.exists() : 
            with open(str(cls.OT_TYPE_FILE), 'r') as f:
                cls.CLS2ID = json.load(f)
                cls.ID2CLS = { v:k for k,v in cls.CLS2ID.items()}
        else:
            print("OT_TYPE setting file doesn't exist.")
            exit()

class Common(BaseAttrs):
    SETTING_FILE = str(project_pth / Path("cfg") / Path("common_settings.json") )
    attrs = [ 
        "temp_pth",
        "track_results_pth"
    ]
    temp_pth : str 
    track_results_pth : str

Parameters.load_OT_type()
Parameters.initiate(module_name=__name__)
Parameters.save_info2json()

Common.initiate(module_name=__name__)
Common.save_info2json()

