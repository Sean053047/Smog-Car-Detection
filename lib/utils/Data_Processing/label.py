import cv2 as cv 
import numpy as np 
from pathlib import Path
import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# sys.path.append(str(Path(__file__).resolve().parent.parent))
ProjectFolder = Path(__file__).resolve().parent
crop_folder = ProjectFolder / Path("Crop_Car")
vids_folder = ProjectFolder / Path("SmogCar")
from __init__ import WITDH_MIN , HEIGHT_MIN
# track_results  = ProjectFolder/ Path("Track_Results")
# from Data_Processing import INTERVAL
import random
from tqdm import tqdm
import json
import shutil
random.seed(13)
np.random.seed(13)


class Label:
    def __init__(self, crop_folder, vids_folder) -> None:
        self.vids_folder = vids_folder
        self.crop_folder = crop_folder
        self.label_folder = str(Path(crop_folder/ "label"))
        Path(self.label_folder).mkdir(parents=True, exist_ok=True)
        self.vids = [str(vid_pth).split('/')[-1] for vid_pth in Path.iterdir(crop_folder)]
        self.vids.remove('label')
        input("Press n: without smog | Press b: with significant smoke | Press s: with ambiguous smoke | Press any other buttons to continue...")
        self.load_labeled() 
        self.load_training()
        self.num_training = len(self.training)
    def label(self):
        for vid in self.vids:
            vid_pth = str(self.vids_folder / Path(vid))+ ".mp4" 
            cap = cv.VideoCapture(vid_pth)
            OTs = [str(OT_folder).split('/')[-1] for OT_folder in Path(self.crop_folder/ Path(vid)).iterdir()]
            
            for OT in OTs:
                imgs_pth = self.random_selected_OTs(vid, OT)
                
                for img_pth in imgs_pth:
                    Done = self.labeled.get(img_pth, None)
                    if Done is not None:
                         continue
                    
                    result = self.label_per_image(cap, img_pth)
                    if result is None:
                        tqdm.write(f"{img_pth} size of image is too small.")
                        continue
                    tqdm.write(f"Label: {img_pth} | {result}")
                    self.update_label(img_pth, result)
                    self.dump_labeled()
                    self.dump_training()
            print(f"{vid}/{OT} is finished.\n")
            cap.release()

    def update_label(self,img_pth, result):
         self.num_training += 1
         new_img_pth = f"{self.num_training}.jpg"
         self.training[new_img_pth] = result
         self.labeled[img_pth] = result
                    
        #  print(self.training)
         Path(self.label_folder).mkdir(parents=True, exist_ok=True)
         shutil.copyfile(img_pth, str(self.label_folder/Path(new_img_pth)) )

    def dump_training(self):
        with open(self.label_folder +'/annotation.json' ,'w') as f :
              json.dump(self.training, f, indent=4,)
              print("Dump training")
    
    def load_training(self):
        if Path(self.label_folder + '/annotation.json').exists():
            print("Load training")
            with open(self.label_folder + '/annotation.json',"r") as f:
                self.training = json.load(f)
        else: 
            print("Create new annotation.json")
            self.training= dict()     
    
    def dump_labeled(self):
         self.labeled_dump = {'/'.join(k.split('/')[-3:]):v for k,v in self.labeled.items()}
         with open(self.label_folder +'/labeled.json' ,'w') as f :
              json.dump(self.labeled_dump, f, indent=4,)
              print("Dump labeled OTs")
    
    def load_labeled(self):
        if Path(self.label_folder + '/labeled.json').exists():
            print("Load labeled OTs")
            with open(self.label_folder + '/labeled.json',"r") as f:
                self.labeled = json.load(f)
                self.labeled = {str(self.crop_folder)+'/'+k:v for k,v in self.labeled.items()}
        else: 
            print("Create new labeled OTs file")
            self.labeled = dict()

    def label_per_image(self, cap, img_pth):
                current = cv.imread(img_pth)
                if current.shape[0] < 100 or current.shape[1] < 100:
                    return None
                
                fid = int(img_pth.split('/')[-1].split('.')[0])
                # fid = 0 if fid <1 else fid
                cap.set(cv.CAP_PROP_POS_FRAMES, fid-1)
                frames = []
                for i in range(fid-2, fid+3):
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                cv.imshow("current label OT image", current)
                i = 0
                while True:
                    cv.imshow("Near frames", frames[i])
                    i = i+1 if i< len(frames)-1 else 0
                    key = cv.waitKey(300)
                    if key == ord('n'):
                        return 0
                    elif key == ord("s"):
                        return 1
                    elif key == ord("b"):
                        return 2  
                    elif key != -1:
                         pass
                
                
    def random_selected_OTs(self, vid, OT):
        vid_OT_folder = self.crop_folder / Path(vid) / Path(OT)
        images = [str(img) for img in Path.iterdir(vid_OT_folder)]
        sample_num = int(len(images)*0.5)
        random_selected = random.sample(images, sample_num)
        if len(random_selected)> 10:
            random_selected = random.sample(images, 10)
             
        return random_selected

if __name__ == "__main__":
    Path.mkdir(crop_folder, parents=True, exist_ok=True)
    a = Label(crop_folder, vids_folder)
    a.label()
    # img = cv.imread('372.jpg')
    # print(img)
    # cv.imshow("img", img)
    # cv.waitKey()
