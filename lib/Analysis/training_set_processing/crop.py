import cv2 as cv 
import numpy as np 
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ByteTrack_yolov7.tracker import Track
from __init__ import FRAME_INTERVAL, OT_EXTEND_RATIO

def crop_vid(tpf:dict[int:list[Track]], vid_pth:str , sv_folder:Path):
    if not Path(vid_pth).exists():
        print("Video doesn't exist!")
        return 
    
    cap = cv.VideoCapture(vid_pth)
    frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    vid_name = vid_pth.split("/")[-1].split('.')[0]
    sv_vid_folder = sv_folder / Path(vid_name)

    for frame_id in range(1, frame_num+1):
        ret, frame = cap.read()
        if not ret :
            print("Read video error.")
            exit()
        if frame_id % FRAME_INTERVAL == 0:
            for track in tpf[frame_id]:
                sv_temp_folder = sv_vid_folder / Path(f"OT_{track.tid}")
                sv_temp_folder.mkdir(parents=True, exist_ok=True)
                
                x1, y1, x2, y2 = tuple(map(int,track.bboxes[frame_id]))
                
                extend_x, extend_y = int(OT_EXTEND_RATIO*(x2-x1)/2), int(OT_EXTEND_RATIO*(y2-y1))
                x1 -= extend_x; x2 += extend_x; y2 += extend_y;
                
                
                x1 = 0 if x1 <0 else x1
                y1 = 0 if y1 <0 else y1
                x2 = WIDTH if x2 > WIDTH else x2
                y2 = HEIGHT if y2 > HEIGHT else y2
                # print(x1, y1, x2,y2)
                cv.imwrite(str(sv_temp_folder/ Path(f"{frame_id}.jpg")), frame[y1:y2, x1:x2 ,:].astype(np.uint8))
                # exit()

if __name__ == "__main__":
    from ByteTrack_yolov7.tracker import  STracks2Tracks, update_tracks_per_frame
    import pickle
    ProjectFolder = Path(__file__).resolve().parent.parent.parent.parent
    Track_Results = ProjectFolder / Path("Track_Results")
    sv_folder = ProjectFolder / Path("data/Crop_Car")

    
    vid_pths = [ str(vid_pth)  for vid_pth in Path.iterdir(ProjectFolder / Path("data/SmogCar"))]

    for vid_pth in vid_pths:
        print("Processing {}".format(vid_pth))
        vid_name =  vid_pth.split("/")[-1].split('.')[0]
        with open(str(Track_Results / Path(f"tracks:{vid_name}.pkl")), "rb") as file:
            tracks = pickle.load(file)
            tracks = STracks2Tracks(tracks)
        with open(str(Track_Results / Path(f"tpf:{vid_name}.pkl")), "rb") as file:
            tracks_per_frame = pickle.load(file)
            tracks_per_frame = update_tracks_per_frame(tracks, tracks_per_frame)

        crop_vid(tracks_per_frame, vid_pth, sv_folder)