import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import json

from config_setting import Parameters

ID2CLS = Parameters.traning_info["label"]
CLS2ID = {v:k for k,v in ID2CLS.items()}
