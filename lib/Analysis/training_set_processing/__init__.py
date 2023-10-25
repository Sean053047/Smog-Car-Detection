from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from analysis_config import Parameters as analysis_par

FRAME_INTERVAL = analysis_par.FRAME_INTERVAL
OT_EXTEND_RATIO = analysis_par.Smoke.OT_EXTEND_RATIO
WITDH_MIN = analysis_par.Smoke.SVM_WIDTH_MIN
HEIGHT_MIN = analysis_par.Smoke.SVM_HEIGHT_MIN
