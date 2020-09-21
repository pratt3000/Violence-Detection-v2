

# Primero instalar openCV package para importar cv2

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# Para descargar los datasets
import download

from random import shuffle

from keras.applications import VGG16

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Activation
import sys


import h5py

in_dir_prueba = 'data'
in_dir = '/home/pratt3000/Downloads/violence-detection-v2'
img_size = 224
img_size_touple = (img_size, img_size)
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 2
_num_files_train = 1
_images_per_file = 20
_num_images_train = _num_files_train * _images_per_file
video_exts = ".avi"
image_model = VGG16(include_top=True, weights='imagenet')
transfer_layer = image_model.get_layer('fc2')
transfer_values_size = K.int_shape(transfer_layer.output)[1]
image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)

chunk_size = 4096
n_chunks = 20
rnn_size = 512

model = Sequential()
model.add(LSTM(rnn_size, input_shape=(n_chunks, chunk_size)))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.load_weights("model.h5")



test_name = "newfi3.avi"
test_names = ["newfi3.avi"]
labels = [0]


def get_frames(current_dir, file_name):
    
    in_file = os.path.join(current_dir, file_name)
    images = []
    vidcap = cv2.VideoCapture(in_file)
    success,image = vidcap.read()    
    count = 0

    while count<_images_per_file:
                
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        res = cv2.resize(RGB_img, dsize=(img_size, img_size),
                                 interpolation=cv2.INTER_CUBIC)
        images.append(res)
    
        success,image = vidcap.read()
        count += 1
        
    resul = np.array(images)
    resul = (resul / 255.).astype(np.float16)
    return resul




def proces_transfer(vid_names, in_dir, labels):

    count = 0
    tam = len(vid_names)
    shape = (_images_per_file,) + img_size_touple + (3,)
    
    while count<tam:
        
        video_name = vid_names[count]
        image_batch = np.zeros(shape=shape, dtype=np.float16)
        image_batch = get_frames(in_dir, test_name)
        
        shape = (_images_per_file, transfer_values_size)
        transfer_values = np.zeros(shape=shape, dtype=np.float16)
        
        transfer_values = \
            image_model_transfer.predict(image_batch)
         
        labels1 = labels[count]
        aux = np.ones([20,2])
        labelss = labels1*aux
        yield transfer_values, labelss
        count+=1



def make_files_validation(n_files):
    
    gen = proces_transfer(test_names, in_dir_prueba, labels)
    numer = 1
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    
    with h5py.File('pruebavalidation.h5', 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]   
            
            numer += 1
  

def process_alldata_validation():
    
    joint_transfer=[]
    frames_num=20
    count = 0
    
    with h5py.File('pruebavalidation.h5', 'r') as f:
            
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch)/frames_num)):
        inc = count+frames_num
        joint_transfer.append([X_batch[count:inc],y_batch[count]])
        count =inc
        
    data =[]
    target=[]
    
    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))
        
    return data, target



make_files_validation(1)
data_val, target_val = process_alldata_validation()
result = model.evaluate(np.array(data_val), np.array(target_val))

print(result)