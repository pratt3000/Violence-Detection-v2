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

    






