import numpy as np
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
from yolov7 import ID2CLS
from .byte_track import STrack
import cv2 as cv

class Track(object):
    def __init__(self, tid, cls_id, start_frame, end_frame, bboxes) -> None:
        self.tid:int = tid
        self.cls_id:int = cls_id
        self.start_frame:int = start_frame
        self.end_frame:int = end_frame
        self.bboxes: dict[np.ndarray] = bboxes

    def __repr__(self) -> str:
        return "OT_{}_{}_({}-{})".format(self.tid, ID2CLS[self.cls_id], self.start_frame, self.end_frame)
    

def STracks2Tracks(Stracks: list[STrack], 
):
    tracks = []
    for t in Stracks:
        tid = t.track_id
        cls_id = t.cls_id
        start_frame = t.start_frame
        end_frame = t.end_frame
        bboxes = t.bboxes
        track = Track(tid, cls_id, start_frame, end_frame, bboxes)
        tracks.append(track)
    
    return tracks

def update_tracks_per_frame(tracks:list[Track], tracks_per_frame) ->dict[int:list[Track]]:
    new_tracks_per_frame = dict()
    for k,v in tracks_per_frame.items():
        new_tracks_per_frame[k] = []
        for tid in v:
            for t in tracks:
                if t.tid == tid:
                    new_tracks_per_frame[k].append(t)
                    break
    return new_tracks_per_frame

def inference(vid_pth, tracks_per_frame, vid_writer):
    def get_color(idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color
    
    video = cv.VideoCapture(vid_pth)
    frame_num = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    for frame_id in range(1, frame_num+1):
        ret, frame = video.read()
        if not ret:
            print("Fail to read images.")
            exit(0)

        for t in tracks_per_frame[frame_id]:
            intbox = tuple(map(int, t.bboxes[frame_id]))
            id_text = '{}: {}'.format(ID2CLS[t.cls_id],int(t.tid))
            color = get_color(abs(t.tid))
            cv.rectangle(frame, intbox[0:2], intbox[2:4], color=color, thickness=3)
            cv.putText(frame, id_text, (intbox[0], intbox[1]), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                        thickness=2)
            
        vid_writer.write(frame)

    
    video.release()