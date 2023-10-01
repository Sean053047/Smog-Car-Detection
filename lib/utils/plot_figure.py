import sys
from pathlib import Path
from matplotlib import pyplot as plt
import pickle
import numpy as np 
# ? append the path of lib
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Analysis.analysis_config import Parameters as analysis_par
from ByteTrack_yolov7.byte_config import Parameters as byte_par
from ByteTrack_yolov7.tracker import Track

def plot_license_conf_bbox_conf(track: Track, vid_name):
    classify_data = track.determine_CarID()

    total_confs = ["license confidence", "bbox confidence", "product of two confidences"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4))
    axes = [ax1, ax2, ax3]
    for i,(k, v) in enumerate(classify_data.items()):
        for q in range(len(total_confs)):
            if q != 2:
                bins = np.arange(v[:,q].min(), 1.01, 0.01)
                axes[q].hist(v[:, q], alpha=1.0, label=k, bins= bins, edgecolor='k')
            else : 
                bins = np.arange((v[:,0]*v[:,1]).min(), 1.01, 0.01)
                axes[q].hist(v[:,0]*v[:,1], alpha=1.0, label=k, bins= bins, edgecolor='k')
                
    for ax, s in zip(axes, total_confs):
        ax.set_title(f"{s}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Times")
        ax.legend()
    plt.tight_layout()
    plt.savefig(byte_par.output_pth / Path(f'{vid_name}:{track}_conf.jpg'), format='jpg')


if __name__ == "__main__":
    vid_pth = "/mnt/HDD-500GB/Smog-Car-Detection/data/SmogCar/SmogCar_15.mp4"
    
    output_pth = Path(byte_par.output_pth)
    vid_name = Path(vid_pth).name.split(".")[0]
    updated_tpf_pth = output_pth / Path(f"updated_tpf:{vid_name}.pkl")
    updated_tracks_pth = output_pth / Path(f"updated_tracks:{vid_name}.pkl")

    if updated_tracks_pth.exists():
        with open(str(updated_tracks_pth), "rb") as file:
            tracks: list[Track] = pickle.load(file)
        with open(str(updated_tpf_pth), "rb") as file:
            tpf = pickle.load(file)
        #     t.determine_CarID()
        plot_license_conf_bbox_conf(tracks, vid_name)