import sys
from pathlib import Path

# ? append the path of lib
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis_config import Parameters as analysis_par

from ByteTrack_yolov7 import Yolov7_Track
from ByteTrack_yolov7.tracker import STracks2Tracks, update_tracks_per_frame, Track, inference
from ByteTrack_yolov7.byte_config import Parameters as byte_par

# ? 1. Import customized Transformer first. At this time, __main__ module knows that transformer.
# ? 2. Then, import predict_image to load module. Because it already knows the customized transformer, it won't raise a AttributeError to load module.
from svm.Transformer import HogTransformer
from svm.predict import predict_image as svm_predict
from sklearn.svm import SVC

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.plots import plot_one_box
from yolov7.utils.general import check_img_size, non_max_suppression

from paddleocr import PaddleOCR

from bbox_process import crop_image, smoke_bbox_extend, calibrate_bbox, box_area_ratio, box_center_distance
import pickle
import threading
import numpy as np
import torch
import cv2

OT_EXTEND_RATIO = analysis_par.Smoke.OT_EXTEND_RATIO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *Load LicenseModel (Yolov7)
license_model = attempt_load(analysis_par.License.model_pth, map_location=device)
stride = int(license_model.stride.max())
imgsz = analysis_par.License.imgsz
imgsz = check_img_size(imgsz, s=stride)
old_img_w = old_img_h = imgsz
old_img_b = 1

# *Load SmokeModel (Yolov7)
smoke_model = attempt_load(analysis_par.Smoke.model_pth, map_location=device)
stride = int(smoke_model.stride.max())
imgsz = analysis_par.Smoke.imgsz
imgsz = check_img_size(imgsz, s=stride)
old_img_w = old_img_h = imgsz
old_img_b = 1


# * Load PaddleOCR setting
ocr = PaddleOCR(use_angle_cls = True, lang='en', use_gpu = True, show_log= False)

def svm_smoke_detect(frame: np.ndarray, frame_id:int, tpf: dict[int:Track]):
    for t in tpf[frame_id]:
        bbox = t.bboxes[frame_id]
        extend_bbox = smoke_bbox_extend(bbox, frame.shape[0], frame.shape[1])
        croped_image = crop_image(frame, extend_bbox)
        pred = svm_predict(croped_image)
        if getattr(t, "svm_preds", None) is None:
            t.svm_preds = dict()
        t.svm_preds[frame_id] = pred

def OCR(frame:np.ndarray, bbox:np.ndarray, bbox_conf:float) -> tuple[str, float]:
    
    cropped = crop_image(image = frame, bbox=bbox)
    results = ocr.ocr(img = cropped, cls=False, det=True, rec=True)[0] # ? iterate result, index= 1 to get string and confidence score
    if results is None:
        return None
    else:
        results = [r[1] for r in results]
        confidences  = np.array([r[1] for r in results])
    
    CarID, license_conf = results[np.argmax(confidences)]
    
    return (CarID, license_conf, bbox_conf)
    
def license_allocate(
    image: torch.tensor, cap_img: np.ndarray, frame_id: int, tpf: dict[int:Track]
):
    global license_model, old_img_h, old_img_w, old_img_b

    cap_height, cap_width = cap_img.shape[:2]
    ratio_height, ratio_width = cap_height / old_img_h, cap_width / old_img_w
    
    frame = torch.from_numpy(image).to(device)
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
        preds = license_model(img, augment=byte_par.augment)[0]
    
    preds = non_max_suppression(preds, byte_par.conf_thresh, byte_par.iou_thresh)[0]
    preds[:, 0] *= ratio_width
    preds[:, 1] *= ratio_height
    preds[:, 2] *= ratio_width
    preds[:, 3] *= ratio_height

    
    valid_bboxes = preds[preds[:, 4] > analysis_par.License.license_conf_thresh, :4]
    valid_confs = preds[preds[:, 4] > analysis_par.License.license_conf_thresh, 4]

    valid_tracks: list[Track] = tpf[frame_id]
    record_bbox = torch.tensor(
        [
            calibrate_bbox(
                t.bboxes[frame_id], height=cap_height, width=cap_width
            )
            for t in valid_tracks
        ],
        device=device,
    )
    area_ratios = box_area_ratio(valid_bboxes[:,:4], record_bbox)
    center_distances = box_center_distance(valid_bboxes, record_bbox)

    C = torch.where(area_ratios>analysis_par.License.area_ratio_thresh, center_distances, float("inf"))
    index_min = torch.argmin(C, dim=1)
    
    for area_ratio, index , bbox, conf, dists in zip(area_ratios, index_min, valid_bboxes, valid_confs, center_distances):
        if bool((area_ratio<analysis_par.License.area_ratio_thresh).all().cpu()):
            continue
        for i, t in enumerate(valid_tracks):
            if i != index:
                t.update_license(frame_id=frame_id)
            else:
                license_info = OCR(frame=cap_img, bbox=bbox.cpu().numpy(), bbox_conf=float(conf.cpu()))
                t.update_license(frame_id= frame_id, dist = float(dists[i].cpu()), license_info=license_info)
                
    for t in  valid_tracks:
        t.reset_dist_record()

def analysis(vid_pth: str, save_inference=True):
    if not Path(vid_pth).exists():
        print("Video didn't exist.")
        return None
    output_pth = Path(byte_par.output_pth)
    vid_name = Path(vid_pth).name.split(".")[0]
    stpf_pth = output_pth / Path(f"stpf:{vid_name}.pkl")
    stracks_pth = output_pth / Path(f"stracks:{vid_name}.pkl")

    if stpf_pth.exists() and stracks_pth.exists():
        with open(str(stracks_pth), "rb") as file:
            stracks = pickle.load(file)
        with open(str(stpf_pth), "rb") as file:
            stpf = pickle.load(file)
    else:
        stracks, stpf = Yolov7_Track(vid_pth, save_inference=False)



    tracks: list[Track] = STracks2Tracks(stracks)
    tpf: dict[int:Track] = update_tracks_per_frame(tracks, stpf)
    # inference(vid_pth, tpf)

    datasets = LoadImages(vid_pth, img_size=imgsz, stride=stride)

    for frame_id, (path, img, cap_img, vid_cap) in enumerate(datasets, 1):
        license_allocate(img, cap_img, frame_id, tpf)


    for t in tracks:
        t.determine_CarID()


    updated_tpf_pth = output_pth / Path(f"updated_tpf:{vid_name}.pkl")
    updated_tracks_pth = output_pth / Path(f"updated_tracks:{vid_name}.pkl")

    with open(str(updated_tracks_pth), 'wb') as file:
        pickle.dump(tracks, file)
    with open(str(updated_tpf_pth), 'wb') as file:
        pickle.dump(tpf, file)
    
    # for t in tracks:
    #     t.determine_CarID()

if __name__ == "__main__":
    vid_pth = "/mnt/HDD-500GB/Smog-Car-Detection/data/SmogCar/SmogCar_15.mp4"
    analysis(vid_pth)