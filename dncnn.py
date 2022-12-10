#DNCNN


#load required packages
import numpy as np
import pandas as pd
from typing import Tuple
import glob
from glob import glob
from pathlib import Path
import os
import keras
from keras import layers
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from skimage import metrics
from keras.models import *
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Lambda,Subtract
from pathlib import Path


#read paired png image data to np array from defined FDCT LDCT path
def read_pair_data(full_path,low_path):
    #load data
    full_list = sorted([f for f in os.listdir(full_path) if f.endswith('.png')])
    low_list = sorted([f for f in os.listdir(low_path) if f.endswith('.png')])

    full = []
    low = []

    for name in full_list:
        img_path = os.path.join(full_path, name ) 
        image = Image.open( img_path ).convert( 'L' ) 
        full.append(np.asarray(image))

    for name in low_list:
        img_path = os.path.join(low_path, name ) 
        image = Image.open( img_path ).convert( 'L' ) 
        low.append(np.asarray(image))

    #store iamges as array
    full = np.asarray(full)
    low = np.asarray(low)
    
    return(full,low)



#normalization
def normal_255(data,factor = 255):

    norm_factor = 255.
    dimension = data.shape[1]
    data = data.astype('float32')/norm_factor
    norm_data = np.reshape(data, (len(data), dimension, dimension, 1))
    return norm_data



#save a list of denoised image as png file
def list_img_save(img_arr_list,img_type,noise_type):    
    path_save_img = result_img / model_name / noise_type
    Path(path_save_img).mkdir(parents=True, exist_ok=True)
    
    for idx,name in enumerate(test_list):
        img = img_arr_list[idx,:,:,0]
        rescaled_img = (np.maximum(img,0)/img.max())*255 # float pixels
        img_obj= Image.fromarray(np.uint8(rescaled_img))
        filename = name[0:-4] + model_name + img_type + ".png"  
        img_obj.save(path_save_img / filename)


#define path of the data
model_name = 'DNCNN'

result_name = "dec05DNCNN"

#define path in a pathlib way
code_path = Path('/Users/jasperzzz/Documents/Projects/LDCT_code/')
data_path = code_path / 'processed/'
result_img = code_path / 'result_img/'


#the path of ata
truth_path_train = data_path /'train/Full/original/'
gau10_path_train = data_path /'train/Low/gau10/'
gau50_path_train = data_path / 'train/Low/gau50/'
poi50_path_train = data_path /'train/Low/poi50/'
poiimg_path_train = data_path /'train/Low/poiimg/'

truth_path_test = data_path /'test/Full/original/'
gau10_path_test = data_path /'test/Low/gau10/'
gau50_path_test = data_path /'test/Low/gau50/'
poi50_path_test = data_path /'test/Low/poi50/'
poiimg_path_test = data_path /'test/Low/poiimg/'

#list of different noisy data
noise_list = ['gau10','gau50','poi50']
noisy_train = [gau10_path_train, gau50_path_train, poi50_path_train]
noisy_test = [gau10_path_test, gau50_path_test, poi50_path_test]



#deine pd dataframe to save the numerical result 
result_name = model_name + "DnCNN_mse" + '.csv'

result_file = code_path / 'result' / result_name
if glob(result_file.as_posix()):
  print("result file exist")
  resultdf = pd.read_csv(result_file)
else:
  print("result file DNE, create new file!")
  column_names = ['model','noise','type','PSNR','MSE']
  resultdf = pd.DataFrame(columns=column_names)




#perform train/test on differet noisy level
for ind,noise_type in enumerate(noise_list):

    #select noisy data
    low_path_train = noisy_train[ind]
    low_path_test = noisy_test[ind]
    test_list = sorted([f for f in os.listdir(low_path_test) if f.endswith('.png')])
    train_full,train_low = read_pair_data(truth_path_train, low_path_train)
    test_full,test_low = read_pair_data(truth_path_test, low_path_test)
    
    #normalization
    train_full_norm = normal_255(train_full)
    train_low_norm = normal_255(train_low)
    test_full_norm = normal_255(test_full)
    test_low_norm = normal_255(test_low)
    
    #definintion of DnCNN
 
    inpt = Input(shape = (None, None, 1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)

    model.compile(optimizer='adam', loss=['mse'])

    validation_split = 0.8
    history = model.fit(train_low_norm, train_full_norm, epochs = 200, batch_size = 16, shuffle = True, validation_split = validation_split)

    #denoise test set
    test_denoised = model.predict(test_low_norm)
    
      # evaluation between original fdct and noisy ldct
    for i in range(test_low_norm.shape[0]):
        #psnr
        PSNR_linear_o = metrics.peak_signal_noise_ratio(test_full_norm[i,:,:,0], test_low_norm[i,:,:,0])
        #mse
        mse_o = metrics.mean_squared_error(test_full_norm[i,:,:,0], test_low_norm[i,:,:,0])
        new_result_df = pd.DataFrame([[model_name,noise_type,'original',PSNR_linear_o,mse_o]], columns=['model','noise','type','PSNR','MSE'])
        resultdf = resultdf.append(new_result_df, ignore_index=True)

        #psnr
        PSNR_linear_d = metrics.peak_signal_noise_ratio(test_full_norm[i,:,:,0],test_denoised[i,:,:,0])
        mse_d = metrics.mean_squared_error(test_full_norm[i,:,:,0],test_denoised[i,:,:,0])
        #mse
        new_result_df = pd.DataFrame([[model_name,noise_type,'denoised',PSNR_linear_d,mse_d]], columns=['model','noise','type','PSNR','MSE'])
        resultdf = resultdf.append(new_result_df, ignore_index=True)
    
    #save result csv
    resultdf.to_csv(result_file, index=False, header=True)
    print("Denoising Result has been saved to " + result_file.as_posix() + ' .')
    
    list_img_save(test_denoised,'_denoised',noise_type)

