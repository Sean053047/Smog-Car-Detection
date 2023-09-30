import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.utils.Base_config import BaseAttrs

class Training_info(BaseAttrs):
    attrs = ["width", "height", "width_min", "height_min", "num_label", "label"]
    width : int
    height : int
    width_min : int 
    height_min : int
    num_label : 2
    label : dict[str: int]
    ID2CLS : dict[int:str]
    CLS2ID : dict[str:int]
class Parameters(BaseAttrs):
    SETTING_FILE = (
        Path(__file__).resolve().parent / Path("cfg") / Path("svm_settings.json")
    )
    attrs = ["Hogify_pth", "Scalify_pth", "Svm_pth", "Training_info"]
    Hogify_pth :str
    Scalify_pth : str
    Svm_pth : str
    Training_info : Training_info
    
    @classmethod
    def update_label(cls):
        cls.Training_info.CLS2ID = cls.Training_info.label
        cls.Training_info.ID2CLS = {v: k for k, v in cls.Training_info.CLS2ID.items()}
        
Parameters.initiate(module_name=__name__)
Parameters.update_label()