import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from svm_config import Parameters
ID2CLS = Parameters.training_info["label"]
CLS2ID = {v:k for k,v in ID2CLS.items()}

