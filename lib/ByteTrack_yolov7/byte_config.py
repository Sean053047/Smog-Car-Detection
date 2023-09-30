import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.utils.Base_config import BaseAttrs
import json
    
class Parameters(BaseAttrs):
    SETTING_FILE = (
        Path(__file__).resolve().parent / Path("cfg") / Path("ByteTrack_settings.json")
    )
    attrs = [
        "output_pth",
        "temp_pth",
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
    OT_TYPE_FILE = Path(__file__).resolve().parent / Path("cfg") / Path("OT-type.json")
    @classmethod
    def load_OT_type(cls):
        if cls.OT_TYPE_FILE.exists() : 
            with open(str(cls.OT_TYPE_FILE), 'r') as f:
                cls.CLS2ID = json.load(f)
                cls.ID2CLS = { v:k for k,v in cls.CLS2ID.items()}
        else:
            print("OT_TYPE setting file doesn't exist.")
            exit()
            
Parameters.load_OT_type()
Parameters.initiate(module_name=__name__)
