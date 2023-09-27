from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tools_config import Parameters

FRAME_INTERVAL = Parameters.FRAME_INTERVAL
OT_EXTEND_RATIO = Parameters.OT_EXTEND_RATIO
WITDH_MIN = Parameters.WIDTH_MIN
HEIGHT_MIN = Parameters.HEIGHT_MIN
