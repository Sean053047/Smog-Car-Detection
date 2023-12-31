import sys
from pathlib import Path

# ? append the path of lib
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis_config import Parameters as analysis_par
from analysis_config import Common
from ByteTrack_yolov7.byte_config import Parameters as byte_par
from ByteTrack_yolov7.tracker import update_tracks_per_frame, inference, combine_tracks
from ByteTrack_yolov7.tracker import STrack, BYTETracker

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
import numpy as np
import torch

ID2CLS = byte_par.ID2CLS
VIDOE_EXT = [".avi", ".mp4", ".mkv"]
OT_EXTEND_RATIO = analysis_par.Smoke.OT_EXTEND_RATIO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *Load LicenseModel (Yolov7)
license_model = attempt_load(analysis_par.License.model_pth, map_location=device)
# *Load SmokeModel (Yolov7)
smoke_model = attempt_load(analysis_par.Smoke.model_pth, map_location=device)
# *Load OT model (Yolov7)
OT_model = attempt_load(byte_par.model_pth, map_location=device)

if int(license_model.stride.max()) != int(smoke_model.stride.max()) != int(OT_model.stride.max()):
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

def yolo_predict(image: torch.tensor, model_type:str):
    global smoke_model, license_model,  old_img_h, old_img_w, old_img_b  
    model = smoke_model if model_type == "smoke" else license_model if model_type == 'license' else OT_model if model_type == "OT" else None
    if model == None:
        print("Wrong Model type.")
        exit()
    var_par = analysis_par.Smoke if model_type == "smoke" else analysis_par.License if model_type == 'license' else byte_par if model_type == "OT" else None
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
        for _ in range(3):
            model(img, augment=augment)[0]
    with torch.no_grad():
        preds = model(img, augment=augment)[0]    
    preds = non_max_suppression(preds, conf_thresh, iou_thresh)[0]
    return preds

def resize_preds_bbox(preds, old_img_h, old_img_w, CAP_HEIGHT, CAP_WIDTH):
    ratio_height, ratio_width = CAP_HEIGHT / old_img_h, CAP_WIDTH / old_img_w
    preds[:, 0] *= ratio_width
    preds[:, 1] *= ratio_height
    preds[:, 2] *= ratio_width
    preds[:, 3] *= ratio_height
    return preds

def svm_smoke_detect(cap_img: np.ndarray, frame_id:int, tracks: list[STrack]) -> None:
    for t in tracks:
        bbox = t.bboxes[frame_id]
        extend_bbox = smoke_bbox_extend(bbox, cap_img.shape[0], cap_img.shape[1], analysis_par.Smoke.OT_EXTEND_RATIO)
        croped_image = crop_image(cap_img, extend_bbox)
        pred = svm_predict(croped_image)
        t.update_smoke_svm_preds(frame_id, pred)

def yolo_smoke_detect(
    image: torch.tensor, cap_img: np.ndarray, frame_id: int, tracks: list[STrack]
):
    CAP_HEIGHT, CAP_WIDTH = cap_img.shape[:2]
    preds = yolo_predict(image, 'smoke')
    preds = resize_preds_bbox(preds, old_img_h, old_img_w, CAP_HEIGHT, CAP_WIDTH)
    record_bbox = torch.tensor([ calibrate_bbox( t.bboxes[frame_id], height=CAP_HEIGHT, width=CAP_WIDTH)for t in tracks],
        device=device,
    )
    
    if preds.shape[0] != 0 and record_bbox.shape[0] != 0:
        center_distances = box_center_distance(preds[:, :4], record_bbox)    
        svm_preds =torch.tensor([t.smoke_svm_preds[frame_id] for t in tracks], device=device)
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
            tracks[indx].update_smoke_yolo_preds(pred=pred.cpu().numpy()[:4], frame_id= frame_id , dist= dists[indx])
        
        for t in  tracks:
            t.reset_smoke_dist()
        return preds, [ tracks[i] for i in indxes ]
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
    image: torch.tensor, cap_img: np.ndarray, frame_id: int, tracks:list[STrack]
): 
    CAP_HEIGHT, CAP_WIDTH = cap_img.shape[:2]
    preds = yolo_predict(image , 'license')
    preds = resize_preds_bbox(preds, old_img_h, old_img_w, CAP_HEIGHT, CAP_WIDTH)

    record_bbox = torch.tensor([calibrate_bbox(t.bboxes[frame_id], height=CAP_HEIGHT, width=CAP_WIDTH) for t in tracks],
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
            tracks[indx].update_license_votes(frame_id= frame_id, dist = float(dists[indx].cpu()), license_info=license_info)
        for t in  tracks:
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
    CAP_HEIGHT, CAP_WIDTH, FRAMES, FPS =  get_video_info(vid_pth)    
    
    tracker = BYTETracker(byte_par, frame_rate = FPS)
    
    if not multi_process:
        datasets = LoadImages(vid_pth, img_size=imgsz, stride=stride)
        for frame_id, (path, img, cap_img, vid_cap) in enumerate(datasets, 1):
            byte_preds= yolo_predict(img , 'OT')      
            current_tracks = tracker.update(
                byte_preds, [CAP_HEIGHT, CAP_WIDTH], (old_img_h, old_img_w)
            )            
            svm_smoke_detect(cap_img, frame_id, current_tracks)
            smoke_preds, OTs = yolo_smoke_detect(image =img, cap_img=cap_img, frame_id=frame_id, tracks = current_tracks)
            license_preds, license_infos = license_allocate(image = img, cap_img = cap_img, frame_id = frame_id , tracks = current_tracks)
    # Todo: Achieve Multiprocessing operation
    else:
        pass

    tracks, tpf_tid = tracker.output_all_tracks(fdif_thresh=10)
    tracks : list[STrack]
    # ? Information allocate
    # updated_tpf_pth = output_pth / Path(f"updated_tpf:{vid_name}.pkl")
    # updated_tracks_pth = output_pth / Path(f"updated_tracks:{vid_name}.pkl")
    # with open(str(updated_tracks_pth), 'wb') as file:
    #     pickle.dump(tracks, file)
    # with open(str(updated_tpf_pth), 'wb') as file:
    #     pickle.dump(tpf_tid, file)

    for t in tracks:
        t.determine_CarID()
        t.determine_Smoke()
    tracks, tpf_tid = combine_tracks(tracks, tpf_tid)

    # ? Dump information pickle files.
    updated_tpf_pth = output_pth / Path(f"updated_tpf:{vid_name}.pkl")
    updated_tracks_pth = output_pth / Path(f"updated_tracks:{vid_name}.pkl")
    with open(str(updated_tracks_pth), 'wb') as file:
        pickle.dump(tracks, file)
    with open(str(updated_tpf_pth), 'wb') as file:
        pickle.dump(tpf_tid, file)

    return tracks, tpf_tid

if __name__ == "__main__":
    # vid_pth = "/mnt/HDD-500GB/Smog-Car-Detection/data/SmogCar/SmogCar_1.mp4"
    # height, width, frame_count, fps = get_video_info(vid_pth)
    import time
    B = True
    if B:
        for i in range(1,17):
            i = 17
            t = time.time()
            print(f"Start : SmogCar_{i}.mp4")
            vid_pth = f"/mnt/HDD-500GB/Smog-Car-Detection/data/SmogCar/SmogCar_{i}.mp4"
            analysis(vid_pth,  False)
            print(f"Finish : SmogCar_{i}.mp4 | Cost time: {time.time()-t}\n")
    else:
        vid_pth = "/mnt/HDD-500GB/Smog-Car-Detection/data/SmogCar/SmogCar_1.mp4"
        output_pth = Path(Common.track_results_pth)
        vid_name = Path(vid_pth).name.split(".")[0]
        updated_tpf_pth = output_pth / Path(f"updated_tpf:{vid_name}.pkl")
        updated_tracks_pth = output_pth / Path(f"updated_tracks:{vid_name}.pkl")
        with open(str(updated_tracks_pth), 'rb') as file:
            tracks: list[STrack] = pickle.load(file)
            t = tracks[6]
            t.determine_CarID()
            t.determine_Smoke()
            # print(t,'|', t.carID, '|', t.license_votes)
            # t.determine_Smoke()
    # with open(str(updated_tpf_pth), 'rb') as file:
    #     tpf_tid = pickle.load( file)
    # print(tracks)
    # print(tpf_tid)