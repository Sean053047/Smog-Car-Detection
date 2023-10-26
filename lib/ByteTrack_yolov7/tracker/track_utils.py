import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from byte_track import STrack
import cv2 as cv

from byte_config import Parameters as byte_par
ID2CLS = byte_par.ID2CLS

def update_tracks_per_frame(tracks:list[STrack], tracks_per_frame:dict[int:list[int]]) ->dict[int:list[STrack]]:
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

def combine_tracks(tracks: list[STrack], tpf_tid: dict[int:list[STrack]], link=False) -> tuple[list[STrack], dict[int:list]]:
    "Based on CarID. If they have same ID, combine them as one."
    new_tracks = []
    for i,t1 in enumerate(tracks):
        if t1.carID != "NULL":    
            for t2 in tracks[i+1:]:
                if t1 != t2 and t1.carID == t2.carID:
                    STrack.combine(t1,t2)
                    # print(f"combine {t1} {t2}")
                    st_f = t2.start_frame+1
                    ed_f = t2.end_frame
                    for fid in range(st_f,ed_f+1):
                        tpf_tid[fid].remove(t2.tid)
                        tpf_tid[fid].append(t1.tid)
                        tpf_tid[fid] = sorted(tpf_tid[fid])
                    tracks.remove(t2)
        new_tracks.append(t1)
    if link :
        return new_tracks, update_tracks_per_frame(new_tracks, tpf_tid)
    else: 
        return new_tracks , tpf_tid