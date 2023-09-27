import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import joblib
import numpy as np

from skimage.transform import resize
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from Transformer import HogTransformer
from config_setting import Parameters

scalify:StandardScaler = joblib.load(Parameters.Scalify_pth)
hogify:HogTransformer = joblib.load(Parameters.Hogify_pth)
svm_clf:svm.SVC = joblib.load(Parameters.Svm_pth)
width = Parameters.training_info["width"]
height = Parameters.training_info["height"]
width_min = Parameters.training_info["width_min"]
height_min = Parameters.training_info["height_min"]


def predict(image:np.ndarray)-> int:
    im_height , im_width = image.shape[:2]
    if im_width < width_min or im_height < height_min:
        return 0
    
    image = resize(image, (height, width))
    hog_im = hogify.transform(np.array([image]))
    scal_im = scalify.transform(hog_im)
    result = svm_clf.predict(scal_im)
    return result[0]

if __name__ == "__main__":
    import cv2 as cv 
    src = Path("/mnt/HDD-500GB/Smog-Car-Detection/data/Crop_Car/label/SmogCar_1")
    for file in Path.iterdir(src):
        image = cv.imread(str(src/file))

        if image.shape[0] < height_min or image.shape[1] < width_min:
            print("Image is too small to predict.")
            continue
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        result = predict(image)
        print(result)
         

