import cv2 as cv
from yolov7.utils.plots import plot_one_box
import torch
import numpy as np 
from pathlib import Path
def plot_boxes(bboxes, img, color=None, labels=None, line_thickness=2):
    if type(bboxes) == torch.Tensor:
        bboxes = bboxes.cpu().numpy()
    if labels is None:
        for bbox in bboxes:
            plot_one_box(bbox, img, color, None, line_thickness)
    else:
        for bbox, label in zip(bboxes, labels):
            plot_one_box(bbox, img, color, str(label), line_thickness)

def get_video_info(vid_pth:str):
    if Path(vid_pth).exists():
        cap  = cv.VideoCapture(vid_pth)
        height, width, num_frame = cap.get(cv.CAP_PROP_FRAME_HEIGHT), cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_COUNT)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        cap.release()
        return height, width, num_frame, fps
    return None

def get_Track_video(vid_pth:str, start_frame , end_frame, label, bboxes, svm_preds, smoke_bboxes, sv_pth:str):
    sv_folder = Path(f"{sv_pth}/Smoke/{Path(vid_pth).stem}")
    sv_folder.mkdir(parents=True, exist_ok=True)
    file_name = Path(f"{label}{Path(vid_pth).resolve().suffix}")
    cap  = cv.VideoCapture(vid_pth)
    height, width, num_frame = cap.get(cv.CAP_PROP_FRAME_HEIGHT), cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    vid_writer = cv.VideoWriter(
            f"{sv_folder/file_name}",
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(width), int(height)),
        )
    for fid in range(start_frame,end_frame+1):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read video. | Start frame: {start_frame} | End frame: {end_frame}")
            exit()
        
        if bboxes.get(fid, None) is not None:
            OT_color = (0,0,255) if svm_preds.get(fid, None) is not None and svm_preds[fid] else (0, 255,0)
            plot_one_box(bboxes[fid],frame, OT_color, label, 2)
        if smoke_bboxes.get(fid, None) is not None:
            
            plot_one_box(smoke_bboxes[fid], frame, (255,0,0), "Smoke", 2)
            cv.imwrite('./smoke_box.jpg', frame)
        vid_writer.write(frame)
    vid_writer.release()
    cap.release()
