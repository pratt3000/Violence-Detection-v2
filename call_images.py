from eval_images import modelDone


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Para descargar los datasets
import download

from random import shuffle
from multiprocessing import Queue

from keras.applications import VGG16

from keras import backend as K
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Activation
import sys

import h5py
import time


ob = modelDone()
file_name = "test/1.avi"
img_size = 224
vidcap = cv2.VideoCapture(file_name)
flag = True

length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

answer = []
start = time.time()

while flag:

    images = np.zeros((20,224,224,3))
    for i in range (0,20):

        success,image = vidcap.read()
        if success==False:
            print("frames over = ",i)
            flag = False
            break

        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(RGB_img, dsize=(img_size, img_size),interpolation=cv2.INTER_CUBIC)
        img = np.array(img)
        img = (img / 255.).astype(np.float16)

        for i in range(0, 19):
            images[i]=images[i+1]
        images[19]=img

    ans = ob.evaluation(images)
    print("ans = ",ans)
    print("total time taken this loop: ", time.time() - start)

    answer.append(ans)

    if flag == False:
        break
    
print(answer)


font = cv2.FONT_HERSHEY_SIMPLEX 
vidcap = cv2.VideoCapture(file_name)
flag = True
frame_width = int(vidcap.get(3)) 
frame_height = int(vidcap.get(4)) 
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('tp1.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 

for i in range(len(answer)):
    if answer[i]==1:
        for i in range(0,20):
            success,frame = vidcap.read()
            if success==False:
                flag = False
                break
            
            cv2.putText(frame,'VIOLENCE', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            result.write(frame) 
    else:
        for i in range(0,20):
            success,frame = vidcap.read()
            if success==False:
                flag = False
                break
            
            cv2.putText(frame,'NON VIOLENCE', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            result.write(frame) 

    if flag == False:
        break
result.release()
            



    






