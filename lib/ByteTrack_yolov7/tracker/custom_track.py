import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from byte_track import STrack
import cv2 as cv

from byte_config import Parameters as byte_par
from string import punctuation
ID2CLS = byte_par.ID2CLS

class Track(object):
    
    
    Lic_BBOX_THRESH = 0.8
    Lic_OCR_THRESH = 0.8

    def __init__(self, tid, cls_id, start_frame, end_frame, bboxes) -> None:
        self.tid:int = tid
        self.cls_id:int = cls_id
        self.start_frame:int = start_frame
        self.end_frame:int = end_frame
        self.bboxes: dict[np.ndarray] = bboxes
        self.carID : str = ''
        
        self.svm_preds : dict[int:int] = dict()
        self.license_votes : dict[ int: (str, float, float)]= dict()  # ? list["CarID", license_conf, bbox_conf]
        
        self.__license_current_dist : float = float("inf")
        self.__license_updated_times: int = 0

    def update_license(self, frame_id:int , dist=None, license_info=None):
        '''update self.license_vote'''
    
        if dist is not None and license_info is not None and dist < self.__license_current_dist :
            self.__license_updated_times += 1
            self.__license_current_dist = dist
            self.license_votes[frame_id] = license_info

    def reset_dist_record(self):
        self.__license_current_dist : float = float("inf")
        self.__license_updated_times = 0
    
    def CarID_format_check(self, carid:str):
        indx = None
        for letter in carid:
            if letter in punctuation:
                indx = carid.find(letter)
                break
        if indx is None:
            return None
        forward_half:str = carid[:indx]
        backward_half:str = carid[indx+1:]
        if (forward_half.isdigit() and backward_half.isalpha()) or (forward_half.isalpha() and backward_half.isdigit()):
            return carid
        else:
            con1 = forward_half.isdigit() and (backward_half.replace('0', 'O')).isalpha()
            con2 = (forward_half.replace('0', 'O')).isalpha() and backward_half.isdigit()
            con3 = forward_half.isalpha() and (backward_half.replace('O', '0')).isdigit()
            con4 = (forward_half.replace('O', '0')).isdigit() and backward_half.isalpha()
            if con1 :
                return forward_half + '-' + backward_half.replace('0', 'O')
            elif con2 :
                return forward_half.replace('0', 'O') + '-' +  backward_half
            elif con3 :
                return forward_half + '-' + backward_half.replace('O', '0')
            elif con4 :
                return forward_half.replace('O', '0') + '-' + backward_half
            
            return None

    def determine_CarID(self):
        if len(self.license_votes) ==0:
            self.carID = "NULL"
            return None
            
        data = np.array([v for v in self.license_votes.values()])
        valid = np.bitwise_and(data[:,1].astype(np.float64) > self.Lic_OCR_THRESH, data[:, 1].astype(np.float64) >self.Lic_BBOX_THRESH)

        data = data[valid, :]
        if data.size == 0:
            self.carID = "NULL"
            return None
        
        carID_list = list()
        for arr in data:
            if arr[0] not in carID_list:
                carID_list.append(arr[0])
        classify_data = dict()
        carID_num = list()
        for s in carID_list:
            classify_data[s] = np.array(data[  data[:,0] == s , :][:,1:], dtype=np.float64)
            carID_num.append(classify_data[s].shape[0])
        carID_num = np.array(carID_num)
        carID_list = np.array(carID_list)
        carID_list = carID_list[np.argsort(carID_num)][::-1]
        
        possible_carID_set = set()
        for id in carID_list:
            possible_carID = self.CarID_format_check(id)
            if possible_carID is not None:
                if possible_carID == id :
                    self.carID = id
                    break
                elif possible_carID in possible_carID_set:
                    self.carID = possible_carID
                    break
                possible_carID_set.add(possible_carID)                
        else:
            self.carID = "NULL"
        return classify_data
    
    def __repr__(self) -> str:
        # if hasattr(self, 'carID'):
        #     return f"{ID2CLS[self.cls_id]: {self.carID}}"
        return "OT_{}_{}_({}-{})".format(self.tid, ID2CLS[self.cls_id], self.start_frame, self.end_frame)
    

def STracks2Tracks(Stracks: list[STrack], filter_out:bool = True, min_showing_times:int = 10):
    '''Goal : Turn STracks to customized Tracks
    par: 
    1.  Stracks: list[STrack]
    2.  filter_out: bool, default = True; 
        When it was true, it will filter out the STrack with showing times less than min_times
    3.  min_times: int, default = 5;
        Means the minimum times which each STrack should have at least min_times.'''
    tracks = []
    for t in Stracks:
        start_frame = t.start_frame
        end_frame = t.end_frame
        if filter_out and (end_frame-start_frame) < min_showing_times:
            continue
        tid = t.track_id
        cls_id = t.cls_id
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
    for i, t in enumerate(tracks,start=1):
        t.tid = i
    
    return new_tracks_per_frame


def inference(vid_pth, tracks_per_frame):
    def get_color(idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color
    

    video = cv.VideoCapture(vid_pth)
    frame_num = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv.CAP_PROP_FPS)
    cap_width, cap_height = video.get(cv.CAP_PROP_FRAME_WIDTH), video.get(cv.CAP_PROP_FRAME_HEIGHT)
    

    vid_writer = cv.VideoWriter(
            byte_par.temp_pth + f"/inference_{Path(vid_pth).name}.mp4",
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(cap_width), int(cap_height)),
        )

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
    vid_writer.release()
    video.release()