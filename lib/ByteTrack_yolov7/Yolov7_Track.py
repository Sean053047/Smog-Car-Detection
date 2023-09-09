import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np 

from yolov7.models.experimental import attempt_load
from yolov7 import DEVICE, ID2CLS
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import (
    check_img_size,
    non_max_suppression,
    set_logging,
)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.logger import setup_logger
from yolov7.utils.datasets import LoadImages
from tracker import BYTETracker, STrack
from loguru import logger

import time

import pickle
from pathlib import Path
import json


class Parameters(enumerate):
    SETTING_FILE = (
        Path(__file__).resolve().parent / Path("cfg") / Path("ByteTrack_settings.json")
    )
    attrs = {
        "output_folder": None,
        "temp_folder" : None,
        "model_pth": None,
        "conf_thresh": None,
        "iou_thresh" : None,
        "track_thresh": None,
        "track_buffer": None,
        "match_thresh": None,
        "min_box_area": None,
        "imgsz": None,
        "augment": None,
        
    }

    @classmethod
    def initiate(cls) -> None:
        UPDATE = False    
        if Path(cls.SETTING_FILE).exists():
            with open(cls.SETTING_FILE, "r") as f:
                settings = json.load(f)
            for k, v in settings.items():
                cls.attrs[k] = v
                setattr(cls, k, v)

            for k, v in cls.attrs.items():
                if not hasattr(cls, k):
                    cls.__input_attr(k)
                    UPDATE = True
        else:
            for attr in cls.attrs.keys():
                cls.__input_attr(attr)
                UPDATE =  True
        if UPDATE:
            with open(cls.SETTING_FILE, "w") as f:
                json.dump(cls.attrs, f, indent=4) 
            print(f"Configure is saved to {cls.SETTING_FILE}")
            exit()

    @classmethod
    def __input_attr(cls, attr):
        while True:
            key = input(f"\rInput {attr} value: ")
            if attr == "output_folder" or attr == "model_pth" or attr == "temp_folder":
                split_list = key.split('/')
                abs_file_pth = Path(__file__).resolve().parent
                if len(split_list) == 1:
                    key = abs_file_pth / Path(split_list[0])
                else:
                    for i, sp in enumerate(split_list):
                        if sp == ".":
                            pass
                        elif sp == "..":
                            abs_file_pth = abs_file_pth.parent
                        else:
                            abs_file_pth = abs_file_pth / Path(sp)
                abs_file_pth = str(abs_file_pth)
                setattr(cls, attr, abs_file_pth)
                cls.attrs[attr] = abs_file_pth
                break
            elif attr == "augment":
                key = True if key == "True" else False if key == "False" else None
                if key == None:
                    print("augment should be True or False")
                    continue
                setattr(cls, attr, key)
                cls.attrs[attr] = key
                break
                    
            if key.replace(".", "").isnumeric():
                key = (
                    int(key)
                    if attr == "min_box_area"
                    or attr == "track_buffer"
                    or attr == "imgsz"
                    else float(key)
                )
                setattr(cls, attr, key)
                cls.attrs[attr] = key
                break
            else:
                print("You should input digits.\n")


VIDOE_EXT = [".avi", ".mp4", ".mkv"]
Parameters.initiate()

output_folder = Path(Parameters.output_folder) 
Path.mkdir(output_folder, exist_ok=True, parents=True)
torch.cuda.empty_cache()
setup_logger(output_folder, filename="val_log.txt")
# Load model
set_logging()
device = DEVICE

model = attempt_load(Parameters.model_pth, map_location=DEVICE)  # load FP32 model

half = device.type != "cpu"  # half precision only supported on CUDA
half = False
if half:
    model.half()
logger.info("Loading model successfully.")

stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, "module") else model.names
imgsz = Parameters.imgsz
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
        if t.tlwh[2] * t.tlwh[3] > Parameters.min_box_area:
            tlwhs.append(t.tlwh)
            tids.append(t.track_id)
            cls_ids.append(t.cls_id)

            
    im = plot_tracking(
        img, tlwhs, tids, cls_ids, frame_id=frame_id + 1
    )
    VideoWriter.write(im)

    

@logger.catch
def Yolov7_Track(vid_pth: str, save_inference = False):
    if not "." + vid_pth.split('.')[-1] in VIDOE_EXT:
        logger.warning("Wrong video type.")
        return []
    else:
        logger.info("Video type is valid.")
    
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
            Parameters.temp_folder + f'/inference_{vid_name}_{timestamp}.mp4', cv.VideoWriter_fourcc(*"mp4v"), fps, (int(CAP_IMG_WIDTH), int(CAP_IMG_HEIGHT))
        )

    frame_id = 0

    tracker = BYTETracker(Parameters, frame_rate=30)
    
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
                model(img, augment=Parameters.augment)[0]
        
        with torch.no_grad():
            pred = model(img, augment = Parameters.augment)[0]
        
        pred = non_max_suppression(pred, Parameters.conf_thresh, Parameters.iou_thresh)[0]
        current_tracks = tracker.update(pred, [CAP_IMG_HEIGHT, CAP_IMG_WIDTH], (old_img_h, old_img_w))
        
        if save_inference:
            inference(cap_img, current_tracks, frame_id, vid_writer)
        frame_id += 1
        if frame_id % 200 == 0:
            logger.info(f"Have processed  {frame_id} frames.")
    
    if save_inference:
        vid_writer.release()

    stracks, stracks_per_frame = tracker.output_all_tracks()
    
    with open(str(output_folder) + f"/tracks_vid:{vid_name}.pkl", "wb") as file :
        pickle.dump(stracks, file)
    with open(str(output_folder) + f'/tracks_per_frame_vid:{vid_name}.pkl', 'wb') as file:
        pickle.dump(stracks_per_frame, file)
    logger.info(f"Save tracks to {output_folder}/tracks_vid:{vid_name}.pkl")
    logger.info(f"Save tracks_per_frame to {output_folder}/tracks_per_frame_vid:{vid_name}.pkl")
    return stracks, stracks_per_frame


if __name__ == "__main__":
    vid_pth = "/mnt/HDD-500GB/Smog-Car-Detection/data/car.mp4"
    Yolov7_Track(vid_pth, True)
    
    with open(str(output_folder) + f"/tracks_vid:car.pkl", "rb") as file :
        Stracks = pickle.load(file)
    with open(str(output_folder) + f'/tracks_per_frame_vid:car.pkl', 'rb') as file:
        tracks_per_frame = pickle.load(file)

    from tracker.custom_track import Track, STracks2Tracks, update_tracks_per_frame
    from tracker.custom_track import inference as cs_inf
    tracks:list[Track] = STracks2Tracks(Stracks)
    tracks_per_frame:dict[int:list[Track]] = update_tracks_per_frame(tracks , tracks_per_frame)
    
    cap = cv.VideoCapture(vid_pth)
    _, image = cap.read()
    CAP_IMG_HEIGHT, CAP_IMG_WIDTH = image.shape[:2] 
    cap.release()
    
    vid_writer = cv.VideoWriter(
            Parameters.temp_folder + f'/inference_test.mp4', cv.VideoWriter_fourcc(*"mp4v"), fps, (int(CAP_IMG_WIDTH), int(CAP_IMG_HEIGHT))
        )
    cs_inf(vid_pth, tracks_per_frame, vid_writer)
    

    