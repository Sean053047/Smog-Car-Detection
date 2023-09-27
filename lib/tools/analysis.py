import threading
from tools_config import Parameters
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ByteTrack_yolov7 import Yolov7_Track
from ByteTrack_yolov7.tracker import STracks2Tracks, update_tracks_per_frame
from svm import predict

OT_EXTEND_RATIO = Parameters.OT_EXTEND_RATIO


def smoke_detect():
    pass


def analysis(vid_pth : str):
    if not Path(vid_pth).exists():
        print("Video didn't exist.")
        return None
    stracks, stpf = Yolov7_Track(vid_pth, save_inference=True)
    tracks = STracks2Tracks(stracks)
    tpf = update_tracks_per_frame(tracks, stpf)

    
    pass
