import sys
from pathlib import Path

# ? append the path of lib
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis_config import Parameters as analysis_par
from analysis_config import Common
from ByteTrack_yolov7 import Yolov7_Track
from ByteTrack_yolov7.tracker import STracks2Tracks, update_tracks_per_frame, Track, inference, combine_tracks

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

from bbox_process import crop_image, smoke_bbox_extend, calibrate_bbox, box_area_ratio, box_center_distance, bbox_diagnol_distance, get_center_point
from lib.utils.video_inference import plot_boxes, get_video_info, get_Track_video
import pickle
import multiprocessing as mp 
import numpy as np
import torch
import cv2

OT_EXTEND_RATIO = analysis_par.Smoke.OT_EXTEND_RATIO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *Load LicenseModel (Yolov7)
license_model = attempt_load(analysis_par.License.model_pth, map_location=device)
# *Load SmokeModel (Yolov7)
smoke_model = attempt_load(analysis_par.Smoke.model_pth, map_location=device)
if int(license_model.stride.max()) != int(smoke_model.stride.max()):
    print("Warning! license model & smoke model have different stride.\n"
          "Can't work together.")
    exit()
stride = int(license_model.stride.max())
imgsz = analysis_par.License.imgsz
imgsz = check_img_size(imgsz, s=stride)
old_img_w = old_img_h = imgsz
old_img_b = 1

# * Load PaddleOCR setting
ocr = PaddleOCR(use_angle_cls = True, lang='en', use_gpu = True, show_log= False)

def yolo_predict(image: torch.tensor, cap_height:int, cap_width:int, model_type:str):
    global smoke_model, license_model,  old_img_h, old_img_w, old_img_b  
    model = smoke_model if model_type == "smoke" else license_model if model_type == 'license' else None
    if model == None:
        print("Wrong Model type.")
        exit()
    
    var_par = analysis_par.Smoke if model_type == "smoke" else analysis_par.License if model_type == 'license' else None
    conf_thresh  = var_par.conf_thresh
    iou_thresh  = var_par.iou_thresh
    augment = var_par.augment
    img = torch.from_numpy(image).to(device).float()
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
            model(img, augment=augment)[0]
    ratio_height, ratio_width = cap_height / old_img_h, cap_width / old_img_w
    with torch.no_grad():
        preds = model(img, augment=augment)[0]    
    preds = non_max_suppression(preds, conf_thresh, iou_thresh)[0]
    preds[:, 0] *= ratio_width
    preds[:, 1] *= ratio_height
    preds[:, 2] *= ratio_width
    preds[:, 3] *= ratio_height
    return preds

def svm_smoke_detect(cap_img: np.ndarray, frame_id:int, tpf: dict[int:Track]) -> None:
    for t in tpf[frame_id]:
        bbox = t.bboxes[frame_id]
        extend_bbox = smoke_bbox_extend(bbox, cap_img.shape[0], cap_img.shape[1], analysis_par.Smoke.OT_EXTEND_RATIO)
        croped_image = crop_image(cap_img, extend_bbox)
        pred = svm_predict(croped_image)
        t.update_smoke_svm_preds(frame_id, pred)


def yolo_smoke_detect(
    image: torch.tensor, cap_img: np.ndarray, frame_id: int, tpf: dict[int:Track]
):
    cap_height, cap_width = cap_img.shape[:2]
    preds = yolo_predict(image, cap_height, cap_width, 'smoke')

    valid_tracks: list[Track] = tpf[frame_id]
    record_bbox = torch.tensor([ calibrate_bbox( t.bboxes[frame_id], height=cap_height, width=cap_width)for t in valid_tracks],
        device=device,
    )
    
    if preds.shape[0] != 0 and record_bbox.shape[0] != 0:
        center_distances = box_center_distance(preds[:, :4], record_bbox)    
        svm_preds =torch.tensor([t.smoke_svm_preds[frame_id] for t in valid_tracks], device=device)
        record_diag = bbox_diagnol_distance(record_bbox).repeat((center_distances.shape[0], 1))
        
        record_points = get_center_point(record_bbox, shift_y= 0.3)
        preds_center = get_center_point(preds)
        below_record = record_points[:,1] < preds_center[:,None,1]
        # ? We care about 
        # ? 1. Is the predicted box closed enought to OT boxes?
        # ? 2. Is the center of predicted boxes below the certain height of OT boxes?
        valids = torch.bitwise_and(center_distances<record_diag, svm_preds!= 0)
        valids = torch.bitwise_and(valids, below_record)
        indxes = torch.where(valids, center_distances, float("inf")).argmin(dim=1)

        for indx , pred, dists, valid in zip(indxes, preds, center_distances, valids):
            if bool(torch.bitwise_not(valid).all().cpu()):
                continue
            valid_tracks[indx].update_smoke_yolo_preds(pred=pred.cpu().numpy()[:4], frame_id= frame_id , dist= dists[indx])
        
        for t in  valid_tracks:
            t.reset_smoke_dist()
        return preds, [ valid_tracks[i] for i in indxes ]
    else:
        return preds, []    
    
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
    
    cap_height, cap_width = cap_img.shape[:2]
    preds = yolo_predict(image, cap_height, cap_width , 'license')
    # print(preds)
    valid_tracks: list[Track] = tpf[frame_id]
    record_bbox = torch.tensor([calibrate_bbox(t.bboxes[frame_id], height=cap_height, width=cap_width) for t in valid_tracks],
        device=device,
    )

    if preds.shape[0] != 0 and record_bbox.shape[0] != 0: 
        area_ratios = box_area_ratio(preds[:, :4], record_bbox)
        center_distances = box_center_distance(preds[:, :4], record_bbox)
        indxes = torch.where(area_ratios>analysis_par.License.area_ratio_thresh, center_distances, float("inf")).argmin(dim =1)

        license_infos:list =[]
        for area_ratio, indx , pred, dists in zip(area_ratios, indxes, preds, center_distances):
            if bool((area_ratio<analysis_par.License.area_ratio_thresh).all().cpu()):
                continue
            license_info = OCR(frame=cap_img, bbox=pred[:4].cpu().numpy(), bbox_conf=float(pred[4].cpu()))
            license_infos.append(license_info)
            valid_tracks[indx].update_license_votes(frame_id= frame_id, dist = float(dists[indx].cpu()), license_info=license_info)
        for t in  valid_tracks:
            t.reset_license_dist()
        return preds, license_infos
    else:
        return preds, []


def analysis(vid_pth: str, multi_process:bool = False):
    if not Path(vid_pth).exists():
        print("Video didn't exist.")
        return None
    output_pth = Path(Common.track_results_pth)
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
    height, width, frame_count, fps = get_video_info(vid_pth)
    tracks: list[Track] = STracks2Tracks(stracks,CUT_FRAME_THRESH=fps*10)
    tpf: dict[int:Track] = update_tracks_per_frame(tracks, stpf)
    
    if not multi_process:
        datasets = LoadImages(vid_pth, img_size=imgsz, stride=stride)
        for frame_id, (path, img, cap_img, vid_cap) in enumerate(datasets, 1):
            svm_smoke_detect(cap_img, frame_id, tpf)
            smoke_preds, OTs = yolo_smoke_detect(image =img, cap_img=cap_img, frame_id=frame_id, tpf = tpf)
            license_preds, license_infos = license_allocate(img, cap_img, frame_id , tpf)
    # Todo: Achieve Multiprocessing operation
    else:
        pass
    # ? Information allocate
    for t in tracks:
        t.determine_CarID()
        t.determine_Smoke()
    tracks, tpf_tid = combine_tracks(tracks, tpf)

    # ? Dump information pickle files.
    updated_tpf_pth = output_pth / Path(f"updated_tpf:{vid_name}.pkl")
    updated_tracks_pth = output_pth / Path(f"updated_tracks:{vid_name}.pkl")
    with open(str(updated_tracks_pth), 'wb') as file:
        pickle.dump(tracks, file)
    with open(str(updated_tpf_pth), 'wb') as file:
        pickle.dump(tpf_tid, file)
    
if __name__ == "__main__":
    vid_pth = "/mnt/HDD-500GB/Smog-Car-Detection/data/SmogCar/SmogCar_15.mp4"
    height, width, frame_count, fps = get_video_info(vid_pth)
    analysis(vid_pth,  False)
    # output_pth = Path(Common.track_results_pth)
    # vid_name = Path(vid_pth).name.split(".")[0]
    # updated_tpf_pth = output_pth / Path(f"updated_tpf:{vid_name}.pkl")
    # updated_tracks_pth = output_pth / Path(f"updated_tracks:{vid_name}.pkl")
    # with open(str(updated_tracks_pth), 'rb') as file:
    #     track = pickle.load(file)
    # with open(str(updated_tpf_pth), 'rb') as file:
    #     tpf = pickle.load( file)