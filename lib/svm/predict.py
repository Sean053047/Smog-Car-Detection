import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import joblib
import numpy as np

from skimage.transform import resize
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from Transformer import HogTransformer
from svm_config import Parameters

scalify:StandardScaler = joblib.load(Parameters.Scalify_pth)
hogify:HogTransformer = joblib.load(Parameters.Hogify_pth)
svm_clf:svm.SVC = joblib.load(Parameters.Svm_pth)
width = Parameters.training_info["width"]
height = Parameters.training_info["height"]
width_min = Parameters.training_info["width_min"]
height_min = Parameters.training_info["height_min"]


def predict(image:np.ndarray)-> int:
    ''' Image : np.ndarray (bgr)'''
    im_height , im_width = image.shape[:2]
    if im_width < width_min or im_height < height_min:
        return 0
    
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    
    image = resize(image, (height, width))
    hog_im = hogify.transform(np.array([image]))
    scal_im = scalify.transform(hog_im)
    result = svm_clf.predict(scal_im)
    return result[0]

if __name__ == "__main__":
    import cv2 as cv 
    import json
    label_folder = Path("/mnt/HDD-500GB/Smog-Car-Detection/data/Crop_Car/label")
    with open(str(label_folder / Path("annotation.json")), "rb") as f :
        labels = json.load(f)        
    gt_smoke = set()
    gt_nosmoke = set()
    pred_smoke = set()
    pred_nosmoke =set()

    for k, v in labels.items():
        if v >= 1:
            gt_smoke.add(k)
        else:
            gt_nosmoke.add(k)

    # print(gt_smoke)
    # print(gt_nosmoke)
    for subdir in Path.iterdir(label_folder):
        if str(subdir).find('SmogCar') == -1:
            continue
        else:
            for file in Path.iterdir(subdir):
                image = cv.imread(str(file))
                gt = labels[file.name]
            
                if image.shape[0] < height_min or image.shape[1] < width_min:
                    print("Image is too small to predict.")
                    continue
                    
                pred = predict(image)
                if pred >= 1 :
                    pred_smoke.add(file.name)
                else:
                    pred_nosmoke.add(file.name)
    tp = gt_smoke & pred_smoke 
    fp = gt_nosmoke & pred_smoke 
    tn = gt_nosmoke & pred_nosmoke
    fn = gt_smoke & pred_nosmoke
    precision = len(tp)/(len(tp)+len(fp))
    recall = len(tp)/ (len(tp)+ len(fn))
    F_measure = 2*precision*recall / (precision+recall)
    print(f"Precision : {precision} | Recall : {recall} | F-measure : {F_measure}")
    n_precision = len(tn)/(len(tn)+len(fn))
    n_recall = len(tn)/ (len(tn)+ len(fp))
    NF_measure = 2*n_precision*n_recall / (n_precision+n_recall)
    
    print(f"N-Precision : {n_precision} | N-Recall : {n_recall} | NF-measure : {NF_measure}")
    
                # time.sleep(3)        
        

