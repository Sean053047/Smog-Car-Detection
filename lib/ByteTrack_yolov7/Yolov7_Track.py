import cv2 as cv
import torch

import numpy as np 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from yolov7.models.experimental import attempt_load
from yolov7 import DEVICE, ID2CLS
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import (
    check_img_size,
    non_max_suppression,
)
from yolov7.utils.datasets import LoadImages
from tracker import BYTETracker, STrack

import time

import pickle
from byte_config import Parameters as byte_par


VIDOE_EXT = [".avi", ".mp4", ".mkv"]

output_pth = Path(byte_par.output_pth) 
Path.mkdir(output_pth, exist_ok=True, parents=True)
torch.cuda.empty_cache()

# Load model
device = DEVICE

model = attempt_load(byte_par.model_pth, map_location=DEVICE)  # load FP32 model

half = device.type != "cpu"  # half precision only supported on CUDA
half = False
if half:
    model.half()

stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, "module") else model.names
imgsz = byte_par.imgsz
imgsz = check_img_size(imgsz, s=stride)
fps = 30 
old_img_w = old_img_h = imgsz
old_img_b = 1

def plot_tracking(image, tlwhs, obj_ids, cls_ids, frame_id=0, fps=0.):
    def get_color(idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

        return color

    im = np.ascontiguousarray(np.copy(image))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    cv.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15  * text_scale)), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for tlwh, obj_id, cls_id  in zip(tlwhs, obj_ids, cls_ids):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        id_text = '{}: {}'.format(ID2CLS[cls_id],int(obj_id))
        color = get_color(abs(obj_id))
        cv.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv.putText(im, id_text, (intbox[0], intbox[1]), cv.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def inference(img: np.ndarray , tracks : list[STrack] ,frame_id: int, VideoWriter):
    tlwhs = []
    tids = []
    cls_ids = []
    for t in tracks:
        t.determine_cls_id()
        if t.tlwh[2] * t.tlwh[3] > byte_par.min_box_area:
            tlwhs.append(t.tlwh)
            tids.append(t.track_id)
            cls_ids.append(t.cls_id)

            
    im = plot_tracking(
        img, tlwhs, tids, cls_ids, frame_id=frame_id + 1
    )
    VideoWriter.write(im)

    

def Yolov7_Track(vid_pth: str, save_inference = False):
    if not "." + vid_pth.split('.')[-1] in VIDOE_EXT:
        return []
    else:
        pass
    global old_img_b, old_img_h, old_img_w

    datasets = LoadImages(vid_pth, img_size=imgsz, stride=stride)
    
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    vid_name = vid_pth.split('/')[-1].split('.')[0]
    
    cap = cv.VideoCapture(vid_pth)
    _, image = cap.read()
    CAP_IMG_HEIGHT, CAP_IMG_WIDTH = image.shape[:2] 
    cap.release()
    
    if save_inference:
        vid_writer = cv.VideoWriter(
            byte_par.temp_pth + f'/inference_{vid_name}.mp4', cv.VideoWriter_fourcc(*"mp4v"), fps, (int(CAP_IMG_WIDTH), int(CAP_IMG_HEIGHT))
        )

    frame_id = 0

    tracker = BYTETracker(byte_par, frame_rate=30)
    
    for path, img, cap_img, vid_cap in datasets:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=byte_par.augment)[0]
        
        with torch.no_grad():
            pred = model(img, augment = byte_par.augment)[0]
        pred = non_max_suppression(pred, byte_par.conf_thresh, byte_par.iou_thresh)[0]
        current_tracks = tracker.update(pred, [CAP_IMG_HEIGHT, CAP_IMG_WIDTH], (old_img_h, old_img_w))
        
        if save_inference:
            inference(cap_img, current_tracks, frame_id, vid_writer)
        frame_id += 1
        if frame_id % 200 == 0:
            print(f"Have processed  {frame_id} frames.")
    
    if save_inference:
        vid_writer.release()

    stracks, stracks_per_frame = tracker.output_all_tracks()
    
    with open(str(output_pth) + f"/tracks:{vid_name}.pkl", "wb") as file :
        pickle.dump(stracks, file)
    with open(str(output_pth) + f'/tpf:{vid_name}.pkl', 'wb') as file:
        pickle.dump(stracks_per_frame, file)
    print(f"Save tracks to {output_pth}/tracks:{vid_name}.pkl")
    print(f"Save tracks_per_frame to {output_pth}/tpf:{vid_name}.pkl")
    return stracks, stracks_per_frame


if __name__ == "__main__":
    
    from tracker import Track, STracks2Tracks, update_tracks_per_frame
    
    vid_pth = "/mnt/HDD-500GB/Smog-Car-Detection/data/SmogCar/SmogCar_1.mp4"
    Yolov7_Track(vid_pth, save_inference=True)
    vid_name = vid_pth.split('/')[-1].split('.')[0]
    with open(str(byte_par.output_pth) + f"/tracks:{vid_name}.pkl", "rb") as file :
        Stracks = pickle.load(file)
    with open(str(byte_par.output_pth) + f'/tpf:{vid_name}.pkl', 'rb') as file:
        tracks_per_frame = pickle.load(file)

    
    tracks:list[Track] = STracks2Tracks(Stracks)
    tracks_per_frame:dict[int:list[Track]] = update_tracks_per_frame(tracks , tracks_per_frame)
    
    