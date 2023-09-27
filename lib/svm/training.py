import joblib
import os
import numpy as np
from skimage.io import imread , imshow
# from cv2 import imread
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.exposure import adjust_gamma

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from Transformer import HogTransformer

import json


import random
class Augmentation():


    @staticmethod
    def horizontal_flip(image_array):
        # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels!
        return image_array[:, ::-1]
    @staticmethod
    def random_blur(image_array):  # blur the image
        return gaussian(image_array, 2, channel_axis=2, mode='reflect')
    
    @staticmethod
    def random_brightness(image_array): # change the brightness randomly
        return adjust_gamma(image_array, gamma=random.uniform(0.8, 1.2), gain=1)

def update_data( data:dict, im, file:str, label:int, imsize:tuple[int, int]):
    width , height = imsize
    image = resize(im, (width, height)) #[:,:,::-1]
    if label != 0:
        data['label'].append(1)
    else:
        data['label'].append(0)

    data['filename'].append(file)
    data['data'].append(image)
    
def resize_all(src, pklname, width=150, height=None, AUG=False, vals=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """

    height = height if height is not None else width

    data = dict()
    data['description'] = 'resized ({0}x{1})SIGN images in rgb'.format(
        int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    with open('./Crop_Car/label/annotation.json', 'r') as f:
        labels_dict = json.load(f)
    pklname = f"{pklname}_{width}px.pkl"
    os.makedirs('./pkl_folder', exist_ok=True)
        
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in src:
        for file in os.listdir(subdir):
            if file.split('.')[-1] in {'jpg', 'png'}:
                im = imread(os.path.join(subdir, file))
                label = labels_dict[file]
                update_data( data, im, file, label, (width,width))
                
                if AUG:
                    num_aug = random.randint(1,3)
                    available_aug = [Augmentation.horizontal_flip, Augmentation.random_brightness, Augmentation.random_blur]
                    for _ in range(num_aug):
                        augment = random.choice(available_aug)
                        available_aug.remove(augment)
                        aug_img = augment(im)
                        update_data( data, aug_img, file, label, (width,width))
        if vals != None:
            joblib.dump(data, f'./pkl_folder/val:{vals}/'+pklname)    
        else:
            joblib.dump(data, './pkl_folder/'+pklname)

    print(f"End of loading {pklname}")
        


if __name__ =='__main__':
    '''IMAGE is RGB'''
    random.seed(32)
    np.random.seed(64)
    width = 150 # ? It is the size of image after resized.
    all_vids = {f"./Crop_Car/label/SmogCar_{i}" for i in range(1,17)}
    txt = open('validation_record.txt', 'w') 
    for _ in range(12):
        training_set = random.sample(all_vids, 12)
        validate_set = [v for v in all_vids if v not in training_set ]
        
        val_list =[]
        for v in validate_set:
            val_list.append(v.split("_")[-1])
        vals = '_'.join(val_list)
        os.makedirs(f"./pkl_folder/val:{vals}")
        print(vals)

        resize_all(src=training_set, pklname="train", width=width, AUG=True, vals=vals)
        resize_all(src=validate_set, pklname="validate", width=width, vals=vals)

        train_data = joblib.load(f'./pkl_folder/val:{vals}/train_{width}px.pkl')
        validate_set = joblib.load(f'./pkl_folder/val:{vals}/validate_{width}px.pkl')


        idx = np.random.permutation(len(train_data['data']))
        X_train, Y_train = np.array(train_data['data'])[idx], np.array(train_data['label'])[idx]
        X_val, Y_val = validate_set['data'], validate_set['label']
        
        hogify = HogTransformer(
            pixels_per_cell=(14, 14),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm='L2-Hys'
        )
        scalify = StandardScaler()

        X_train_hog = hogify.fit_transform(X_train)
        X_train_prepared = scalify.fit_transform(X_train_hog)


        svm_clf = svm.SVC()
        svm_clf.fit(X_train_prepared, Y_train)

        X_val_hog = hogify.transform(X_val)
        X_val_prepared = scalify.transform(X_val_hog)
        Y_pred = svm_clf.predict(X_val_prepared)

        # print(np.array(y_pred == y_test)[:25])
        print('')
        validation_score = 100*np.sum(Y_pred == Y_val)/len(Y_val)
        
        print('Percentage correct: ', validation_score)
        txt.write(f'val:{vals}_Percentage correct: {validation_score}\n')

        joblib.dump(svm_clf, f'./pkl_folder/val:{vals}/SmokeModel_{width}.pkl')
        joblib.dump(hogify, f'./pkl_folder/val:{vals}/SmokeHogify_{width}.pkl')
        joblib.dump(scalify, f'./pkl_folder/val:{vals}/SmokeScalify_{width}.pkl')
    txt.close()