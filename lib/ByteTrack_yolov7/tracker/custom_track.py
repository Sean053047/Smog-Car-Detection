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
    
    Lic_BBOX_THRESH = 0.6
    Lic_OCR_THRESH = 0.6
    SVM_ACCEPTABLE_MISS = 5
    CUT_FRAME_THRESH : int

    def __init__(self, tid, cls_id, start_frame, end_frame, bboxes, CUT_FRAME_THRESH=300) -> None:
        self.tid:int = tid
        self.cls_id:int = cls_id
        self.start_frame:int = start_frame
        self.end_frame:int = end_frame
        self.bboxes: dict[np.ndarray] = bboxes
        self.carID : str = ''
        self.smoke_start_frame : int = None
        self.smoke_end_frame : int = None
        self.is_smoke : bool = False

        self.smoke_svm_preds : dict[int:int] = dict()
        self.smoke_yolo_preds: dict[int:np.ndarray] = dict()
        self.license_votes : dict[ int: (str, float, float)]= dict()  # ? list["CarID", license_conf, bbox_conf]
        
        self.__license_current_dist : float = float("inf")
        self.__smoke_current_dist : float = float("inf")

        self.CUT_FRAME_THRESH = CUT_FRAME_THRESH

    def update_smoke_svm_preds(self, frame_id: int, pred: np.ndarray) -> None:
        self.smoke_svm_preds[frame_id] = pred

    def update_smoke_yolo_preds(self, frame_id:int, pred: np.ndarray=None, dist=None) -> None:
        if dist is not None and pred is not None and dist < self.__smoke_current_dist:
            self.__smoke_current_dist = dist
            self.smoke_yolo_preds[frame_id] = pred
    
    def update_license_votes(self, frame_id:int , dist=None, license_info=None):
        '''update self.license_vote'''
        if dist is not None and license_info is not None and dist < self.__license_current_dist :
            self.__license_current_dist = dist
            self.license_votes[frame_id] = license_info

    def reset_license_dist(self) -> None:
        self.__license_current_dist : float = float("inf")
    
    def reset_smoke_dist(self) -> None:
        self.__smoke_current_dist : float = float("inf")

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
        if (forward_half.isdigit() and backward_half.isalpha()):
            return f"{forward_half}-{backward_half.upper()}"
        elif (forward_half.isalpha() and backward_half.isdigit()):
            return f"{forward_half.upper()}-{backward_half}"
        else:
            con1 = forward_half.isdigit() and (backward_half.replace('0', 'O')).isalpha()
            con2 = (forward_half.replace('0', 'O')).isalpha() and backward_half.isdigit()
            con3 = forward_half.isalpha() and (backward_half.replace('O', '0')).isdigit()
            con4 = (forward_half.replace('O', '0')).isdigit() and backward_half.isalpha()
            if con1 :
                return f"{forward_half}-{backward_half.replace('0', 'O').upper()}"
            elif con2 :
                return f"{forward_half.replace('0', 'O').upper()}-{backward_half}"
            elif con3 :
                return f"{forward_half.upper()}-{backward_half.replace('O', '0')}"
            elif con4 :
                return f"{forward_half.replace('O', '0')}-{backward_half.upper()}"
            
            return None

    def determine_CarID(self):
        if len(self.license_votes) ==0:
            self.carID = "NULL"
            return None
            
        data = np.array([v for v in self.license_votes.values()])
        valid = np.bitwise_and(data[:,1].astype(np.float64) > self.Lic_OCR_THRESH, data[:, 2].astype(np.float64) >self.Lic_BBOX_THRESH)

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
    
    def determine_Smoke(self):
        svm_preds = [v for v in self.smoke_svm_preds.values()]
        yf_indx = np.array([k for k in self.smoke_yolo_preds.keys()])
        
        if len(self.smoke_yolo_preds) == 0:
            if (np.array(svm_preds)<1).all():
                return
            svm_indx, svm_chg_indx = self.get_svm_indx_chg()
            bst_indx = 0
            if len(svm_chg_indx) == 1:
                if (svm_chg_indx[bst_indx] - svm_indx[0])/self.CUT_FRAME_THRESH < 0.1:
                    return
                center_indx = int((svm_chg_indx[0]+svm_indx[0])/2)
                start_frame = center_indx - int(self.CUT_FRAME_THRESH/2)
                end_frame = center_indx + int(self.CUT_FRAME_THRESH/2)+1
                self.is_smoke = True
                self.smoke_start_frame, self.smoke_end_frame = self.compare_frames(start_frame, end_frame)
            else:
                frame_diff =0
                for i in range(1,len(svm_chg_indx),2):
                    print(svm_chg_indx)
                    print(self, len(svm_chg_indx), i,' ', end="")
                    diff = svm_chg_indx[i] - svm_chg_indx[i-1]
                    bst_indx = i if diff > frame_diff else bst_indx
                    frame_diff = diff if diff > frame_diff else frame_diff
                # ? If without yolo smoke and the number of svm smoke frames / CUT_FRAME_THRESH is less than 0.1
                # ? Ignore
                if frame_diff/self.CUT_FRAME_THRESH < 0.1:
                    return
                else:
                    # Todo: finish 
                    center_indx = int((svm_chg_indx[bst_indx]+svm_chg_indx[bst_indx+1])/2)
                    start_frame = center_indx - int(self.CUT_FRAME_THRESH/2)
                    end_frame = center_indx + int(self.CUT_FRAME_THRESH/2)+1
                    self.is_smoke = True
                    self.smoke_start_frame, self.smoke_end_frame = self.compare_frames(start_frame, end_frame)
            return
        else:
            yf_bbox_area = np.array([((v[2] - v[0]) * (v[3] - v[1])) for v in self.smoke_yolo_preds.values()])
            bst_area_indx = yf_indx[np.argmax(yf_bbox_area)]
            svm_indx, svm_chg_indx = self.get_svm_indx_chg()
            if len(svm_chg_indx) == 0 :
                if svm_preds[0]==1: 
                    start_frame = bst_area_indx - int(self.CUT_FRAME_THRESH/2)
                    end_frame = bst_area_indx + int(self.CUT_FRAME_THRESH/2)+1
                    self.is_smoke = True
                    self.smoke_start_frame, self.smoke_end_frame = self.compare_frames(start_frame, end_frame)
                return
            
            svm_chg_indx.append(bst_area_indx)
            svm_chg_indx = sorted(svm_chg_indx)
            v = svm_chg_indx.index(bst_area_indx)
            # * bst_area_indx is the last element of svm_chg_indx
            var1 = svm_indx[0] if v ==0 else svm_chg_indx[v-1]
            var2 = svm_indx[-1] if v == len(svm_chg_indx)-1 else svm_chg_indx[v+1]
            before_ratio = (svm_chg_indx[v] - var1) / (var2 - var1)
            after_ratio = (var2 - svm_chg_indx[v]) / (var2 - var1)
            # print(self.CUT_FRAME_THRESH,before_ratio, after_ratio)
            start_frame = bst_area_indx - int(self.CUT_FRAME_THRESH * before_ratio)
            end_frame = bst_area_indx + int(self.CUT_FRAME_THRESH * after_ratio)
            self.is_smoke = True
            self.smoke_start_frame, self.smoke_end_frame = self.compare_frames(start_frame, end_frame)

    def get_svm_indx_chg(self):
        # ? Record the index when t.smoke_svm_preds[t] changes
        svm_preds = [v for v in self.smoke_svm_preds.values()]
        svm_indx = np.array([k for k in self.smoke_svm_preds.keys()])
        chg_indx =[]
        for i, p in enumerate(svm_preds):
            if i != 0 and p == abs(svm_preds[i-1] -1):
                chg_indx.append(i)
        svm_chg_indx = svm_indx[chg_indx].tolist()
        # ? If there're just less than 5 frames predected 0, ignore it.
        st_indx = 1 if svm_preds[0] >=1 else 2
        
        new_svm_chg_indx = []
        for i, v in enumerate(svm_chg_indx):
            new_svm_chg_indx.append(v)
            if i < st_indx :
                continue
            if (svm_chg_indx[i-1] in new_svm_chg_indx  and v - svm_chg_indx[i-1] < self.SVM_ACCEPTABLE_MISS):
                new_svm_chg_indx.remove(svm_chg_indx[i-1])
                new_svm_chg_indx.remove(svm_chg_indx[i])
        
        return svm_indx, new_svm_chg_indx
    
    def compare_frames(self, start_frame:int, end_frame:int):
        st_b = False
        ed_b = False
        if start_frame < self.start_frame:
            start_frame = self.start_frame 
            st_b = True
        if  end_frame > self.end_frame:
            end_frame = self.end_frame 
            ed_b = True
        diff = end_frame -start_frame +1
        if diff < self.CUT_FRAME_THRESH:    
            if st_b and not ed_b:
                end_frame += self.CUT_FRAME_THRESH - diff
            elif ed_b and not st_b:
                start_frame -= self.CUT_FRAME_THRESH-diff 
            elif st_b and ed_b:
                start_frame -= int((self.CUT_FRAME_THRESH - diff)/2)
                end_frame += int((self.CUT_FRAME_THRESH - diff)/2)
    
        start_frame = self.start_frame if start_frame < self.start_frame else start_frame
        end_frame = self.end_frame if end_frame > self.end_frame else end_frame

        return start_frame , end_frame
                
    def __repr__(self) -> str:
        # if hasattr(self, 'carID'):
        #     return f"{ID2CLS[self.cls_id]: {self.carID}}"
        return "OT_{}_{}_({}-{})".format(self.tid, ID2CLS[self.cls_id], self.start_frame, self.end_frame)
    
    @staticmethod
    def combine(track1, track2):
        '''Combine track2 information to track1. If track1 has correspond info, it will skip that information.'''
        track1.start_frame = track2.start_frame if track1.start_frame > track2.start_frame else track1.start_frame
        track1.end_frame = track2.end_frame if track1.end_frame < track2.end_frame else track1.end_frame
        for k,v in track2.bboxes.items():
            if track1.bboxes.get(k, None) is not None:
                continue
            track1.bboxes[k] = v 
        for k,v in track2.smoke_yolo_preds.items():
            if track1.smoke_yolo_preds.get(k, None) is not None:
                continue
            track1.smoke_yolo_preds[k] = v
        for k,v in track2.smoke_svm_preds.items():
            if track1.smoke_svm_preds.get(k, None) is not None:
                continue
            track1.smoke_svm_preds[k] = v
        if track1.carID == "":
            track1.carID = track2.carID
        track1.determine_Smoke()
        

def STracks2Tracks(Stracks: list[STrack], filter_out:bool = True, min_showing_times:int = 10, CUT_FRAME_THRESH=300):
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
        track = Track(tid, cls_id, start_frame, end_frame, bboxes, CUT_FRAME_THRESH)
        tracks.append(track)
    return tracks

def update_tracks_per_frame(tracks:list[Track], tracks_per_frame:dict[int:list[int]]) ->dict[int:list[Track]]:
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
            byte_par.temp_pth + f"/MOT_{Path(vid_pth).name}",
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

def combine_tracks(tracks: list[Track], tpf: dict[int:list[Track]]) -> tuple[list[Track], dict[int:list[Track]]]:
    "Based on CarID. If they have same ID, combine them as one."
    tpf_tid = {k:[t.tid for t in v] for k, v in tpf.items()}
    new_tracks = []
    for i,t1 in enumerate(tracks):
        if t1.carID != "NULL":    
            for t2 in tracks[i+1:]:
                if t1 != t2 and t1.carID == t2.carID:
                    Track.combine(t1,t2)
                    # print(f"combine {t1} {t2}")
                    st_f = t2.start_frame+1
                    ed_f = t2.end_frame
                    for fid in range(st_f,ed_f+1):
                        tpf_tid[fid].remove(t2.tid)
                        tpf_tid[fid].append(t1.tid)
                        tpf_tid[fid] = sorted(tpf_tid[fid])
                    tracks.remove(t2)
        new_tracks.append(t1)
                
    return new_tracks, update_tracks_per_frame(new_tracks, tpf_tid)