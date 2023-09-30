import sys
from pathlib import Path

# ? append the path of lib
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis_config import Parameters as analysis_par

from ByteTrack_yolov7 import Yolov7_Track
from ByteTrack_yolov7.tracker import STracks2Tracks, update_tracks_per_frame, Track
from ByteTrack_yolov7.byte_config import Parameters as byte_par

# ? 1. Import customized Transformer first. At this time, __main__ module knows that transformer.
# ? 2. Then, import predict_image to load module. Because it already knows the customized transformer, it won't raise a AttributeError to load module.
from svm.Transformer import HogTransformer
from svm.predict import predict_image as svm_predict
from sklearn.svm import SVC

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.plots import plot_one_box
from yolov7.utils.general import (
    check_img_size,
    non_max_suppression,
    box_diou
)

from bbox_process import crop_image, smoke_bbox_extend
import pickle
import threading
import numpy as np
import torch

OT_EXTEND_RATIO = analysis_par.Smoke.OT_EXTEND_RATIO

# *Load LicenseModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(analysis_par.License.model_pth)

license_model = attempt_load(analysis_par.License.model_pth, map_location=device)
stride = int(license_model.stride.max())
imgsz = analysis_par.License.imgsz
imgsz = check_img_size(imgsz, s=stride)
old_img_w = old_img_h = imgsz
old_img_b = 1
import cv2

def smoke_detect(frame: np.ndarray, frame_id:int, tpf: dict[int:Track]):
    for t in tpf[frame_id]:
        bbox = t.bboxes[frame_id]
        extend_bbox = smoke_bbox_extend(bbox, frame.shape[0], frame.shape[1])
        croped_image = crop_image(frame, extend_bbox)
        pred = svm_predict(croped_image)
        if getattr(t, "svm_preds", None) is None:
            t.svm_preds = dict()
        t.svm_preds[frame_id] = pred


def license_allocate(frame: np.ndarray, CAP_IMG_SIZE, frame_id:int , tpf:dict[int:Track]):

    global license_model, old_img_h, old_img_w, old_img_b
    
    ratio_height, ratio_width = CAP_IMG_SIZE[0]/old_img_h, CAP_IMG_SIZE[1]/ old_img_w
    frame = torch.from_numpy(frame).to(device)
    img = frame.float()
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
            license_model(img, augment=byte_par.augment)[0]

    with torch.no_grad():
        pred = license_model(img, augment=byte_par.augment)[0]
    pred = non_max_suppression(pred, byte_par.conf_thresh, byte_par.iou_thresh)[0]
    
    pred[:,0] *= ratio_width
    pred[:,1] *= ratio_height
    pred[:,2] *= ratio_width
    pred[:,3] *= ratio_height

    valid_pred_range = pred[:,4] > analysis_par.License.conf_thresh
    valid_pred = pred[valid_pred_range ,  : ]

    valid_tracks:list[Track] = tpf[frame_id]
    # print(valid_tracks)
    # for license_bbox in valid_pred:
    #     print(license_bbox)
    #     for t in valid_tracks:
    #         print(t.bboxes[frame_id])
    # exit()

    return pred

def analysis(vid_pth: str, save_inference=True):
    if not Path(vid_pth).exists():
        print("Video didn't exist.")
        return None
    output_pth = Path(byte_par.output_pth)
    vid_name = Path(vid_pth).name.split(".")[0]
    stpf_pth = Path(output_pth / Path(f"stpf:{vid_name}.pkl"))
    stracks_pth = output_pth / Path(f"stracks:{vid_name}.pkl")
    
    if stpf_pth.exists() and stracks_pth.exists():
        with open(str(stracks_pth), "rb") as file:
            stracks = pickle.load(file)
        with open(str(stpf_pth), "rb") as file:
            stpf = pickle.load(file)
    else:
        stracks, stpf = Yolov7_Track(vid_pth, save_inference)

    tracks: list[Track] = STracks2Tracks(stracks)
    tpf: dict[int:Track] = update_tracks_per_frame(tracks, stpf)

    cap = cv2.VideoCapture(vid_pth)
    CAP_HEIGHT, CAP_WIDTH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()


    if save_inference:
        vid_writer = cv2.VideoWriter(
            byte_par.temp_pth + f"/inference_license_{vid_name}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (int(CAP_WIDTH), int(CAP_HEIGHT)),
        )
    datasets = LoadImages(vid_pth, img_size=imgsz, stride=stride)

    for frame_id, (path, img, cap_img, vid_cap) in enumerate(datasets, 1):
        pred = license_allocate(img, (CAP_HEIGHT, CAP_WIDTH), frame_id, tpf)

        for p in pred:
            if p[4] > 0.5:
                plot_one_box(p, img=cap_img, color =(255,0,0), label=str(p[4]),line_thickness=3)
        vid_writer.write(cap_img)
    vid_writer.release()
        # exit()



        
    # print(t.svm_preds)


if __name__ == "__main__":
    vid_pth = "/mnt/HDD-500GB/Smog-Car-Detection/data/SmogCar/SmogCar_15.mp4"
    analysis(vid_pth)
