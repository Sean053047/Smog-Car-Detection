import sys
from pathlib import Path 
# ? append the path of lib
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tools_config import Parameters as tool_par

from ByteTrack_yolov7 import Yolov7_Track
from ByteTrack_yolov7.tracker import STracks2Tracks, update_tracks_per_frame, Track
from ByteTrack_yolov7.byte_config import Parameters as byte_par

# ? 1. Import customized Transformer first. At this time, __main__ module knows that transformer.
# ? 2. Then, import predict_image to load module. Because it already knows the customized transformer, it won't raise a AttributeError to load module.
from svm.Transformer import  HogTransformer
from svm.predict import predict_image as svm_predict
from sklearn.svm import SVC

from bbox_process import crop_image
import pickle
import cv2 as cv 
import threading
import numpy as np

OT_EXTEND_RATIO = tool_par.OT_EXTEND_RATIO


def smoke_detect(frame:np.ndarray, frame_id , tpf:dict[int:Track]):
    # print(tpf[109])
    for t in tpf[frame_id]:
        # if t.tid == 2:
        #     print(t, " | ", frame_id)
        bbox = t.bboxes[frame_id]
        croped_image = crop_image(frame, bbox)
        pred = svm_predict(croped_image)
        if getattr(t, 'svm_preds', None) is None:
            t.svm_preds = dict()
        t.svm_preds[frame_id] = pred


def license_allocate():
    pass

def analysis(vid_pth : str, save_inference= True):
    if not Path(vid_pth).exists():
        print("Video didn't exist.")
        return None
    output_pth = Path(byte_par.output_pth)
    vid_name = Path(vid_pth).name.split(".")[0]
    # print(Path(output_pth/ Path(f"stpf:{vid_name}.pkl")).exists())
    # print(Path(output_pth/ Path(f"stracks:{vid_name}.pkl")).exists())
    # exit()
    
    if Path(output_pth/ Path(f"stpf:{vid_name}.pkl")).exists() and Path(output_pth/ Path(f"stracks:{vid_name}.pkl")).exists():
        with open(str(byte_par.output_pth) + f"/stracks:{vid_name}.pkl", "rb") as file :
            stracks = pickle.load(file)
        with open(str(byte_par.output_pth) + f'/stpf:{vid_name}.pkl', 'rb') as file:
            stpf = pickle.load(file)
    else:
        stracks, stpf = Yolov7_Track(vid_pth, save_inference)
        
    tracks:list[Track] = STracks2Tracks(stracks)
    tpf:dict[int:Track] = update_tracks_per_frame(tracks, stpf)
    
    cap = cv.VideoCapture(vid_pth)
    FRAME_LEN = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    HEIGHT, WIDTH = cap.get(cv.CAP_PROP_FRAME_HEIGHT), cap.get(cv.CAP_PROP_FRAME_WIDTH)
    

    for frame_id in range(1, FRAME_LEN+1):
        ret, frame = cap.read()
        if not ret: 
            print("Fail to read video.")
            return None
        smoke_detect(frame, frame_id, tpf)

    cap.release()


        # print(t.svm_preds)

if __name__ == "__main__":
    
    vid_pth = "/mnt/HDD-500GB/Smog-Car-Detection/data/SmogCar/SmogCar_8.mp4"
    analysis(vid_pth)
