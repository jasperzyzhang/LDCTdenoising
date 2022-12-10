#convert dicom CT image to jpg format and downsample to 200pix
# load required tools
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from typing import Tuple
from PIL import Image
import glob
from pathlib import Path
import os


#downsample by averaging
def downsample_by_averaging(img: np.ndarray, window_shape: Tuple[int, int]) -> np.ndarray:
    return np.mean(
        img.reshape((
            *img.shape[:-2],
            img.shape[-2] // window_shape[-2], window_shape[-2],
            img.shape[-1] // window_shape[-1], window_shape[-1],
        )),
        axis=(-1, -3),
    )



data_path =  '/Users/jasperzzz/Library/CloudStorage/OneDrive-UniversityofToronto/BiostatPhD/Year1/Fall&Y_courses/CSC2529ComputImage/data/'
dicom_sample_path = data_path + "imgsamples/"
patient_list = [f for f in os.listdir(dicom_sample_path) if f.endswith('img')]


#randomly split patient into training testing sets.
import random
from random import sample
random.seed(1997)
patient_train = sorted(sample(sorted(patient_list),round(len(patient_list)*0.8)))
patient_test = sorted(list(set(patient_list) - set(patient_train)))



#1 read dicom image
#2 downsample and cut to 200pix
#3 save the image
#4 add noise (gaussian, poisson) and save image to png file
dataset_list = ['train','test']

for dataset in dataset_list:
    if dataset == 'train':
        p_list = patient_train
    else:
        p_list = patient_test

    for patient in p_list:
        patient_id = patient    

        for type_ct in ["Full","Low"]:
            path_save = data_path + "/processed/" + dataset + "/" + type_ct +"/"
            Path(path_save).mkdir(parents=True, exist_ok=True)

            path_save_original = data_path + "/processed/" + dataset + "/" + type_ct +"/original/"
            path_save_gau10 = data_path + "/processed/" + dataset + "/" + type_ct +"/gau10/"
            path_save_gau50 = data_path + "/processed/" + dataset + "/" + type_ct +"/gau50/"
            path_save_poi50 = data_path + "/processed/" + dataset + "/" + type_ct +"/poi50/"
            path_save_poiimg = data_path + "/processed/" + dataset + "/" + type_ct +"/poiimg/"

            Path(path_save_original).mkdir(parents=True, exist_ok=True)
            Path(path_save_gau10).mkdir(parents=True, exist_ok=True)
            Path(path_save_gau50).mkdir(parents=True, exist_ok=True)
            Path(path_save_poi50).mkdir(parents=True, exist_ok=True)
            Path(path_save_poiimg).mkdir(parents=True, exist_ok=True)
            
            fdpath = dicom_sample_path + patient + "/*"+ type_ct + "*"

            for f in glob.iglob(fdpath): # generator, search immediate subdirectories 
                
                img_files = sorted(os.listdir(f))
                middleIndex = int((len(img_files) - 1)/2) # find middle 20 images
                middle_files = img_files[(middleIndex-10):(middleIndex + 10)]
                for idx,image in enumerate(middle_files):
                    filename = f + "/" + image
                    ds = pydicom.dcmread(filename) #read image
                    img = ds.pixel_array.astype(float)
                    rescaled_img = (np.maximum(img,0)/img.max())*255 # float pixels
             
                    img_org = np.uint8(rescaled_img) #integers pixels
                    img_down = np.uint8(downsample_by_averaging(img_org, (2, 2))) # 128 (4,4), 256(2,2)
                    img_down = img_down[28:228,28:228]
                    img_down_arr= Image.fromarray(img_down)
                    img_down_arr.save(path_save_original + patient_id + "_pix200_" + type_ct + "_"  + str(idx + 1)  + ".png")
      

                    #gaussian(0,10)
                    gau10 = np.random.normal(0,10,img_down.shape)
                    noisy_gau10 = np.uint8(np.clip(img_down + gau10,0,255))
                    img_gau10_arr= Image.fromarray(noisy_gau10)
                    img_gau10_arr.save(path_save_gau10 + patient_id + "_pix200_" + type_ct + "_"  + str(idx + 1)  + ".png")
                    
   
                    #gaussian(0,50)
                    gau50 = np.random.normal(0,50,img_down.shape)
                    noisy_gau50 = np.uint8(np.clip(img_down + gau50,0,255))
                    img_gau50_arr= Image.fromarray(noisy_gau50)

                    img_gau50_arr.save(path_save_gau50 + patient_id + "_pix200_" + type_ct + "_"  + str(idx + 1)  + ".png")
                    
                    
                    #poisson skimage
                    vals = len(np.unique(img_down))
                    vals = 2 ** np.ceil(np.log2(vals))
                    # Generating noise for each unique value in image.
                    noisy_poiimg = np.uint8(np.clip(np.random.poisson(img_down * vals) / float(vals),0,255))
                    img_poiimg_arr= Image.fromarray(noisy_poiimg)
                    img_poiimg_arr.save(path_save_poiimg + patient_id + "_pix200_" + type_ct + "_"  + str(idx + 1)  + ".png")

                    
                    #poisson lambda 50
                    poi50 = np.random.poisson(lam = 50, size = img_down.shape)
                    noisy_poi50 = np.uint8((np.clip(img_down + poi50,0,255)))
                    img_poi50_arr= Image.fromarray(noisy_poi50)
                    img_poi50_arr.save(path_save_poi50 + patient_id + "_pix200_" + type_ct + "_"  + str(idx + 1)  + ".png")

                    

