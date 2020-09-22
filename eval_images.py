import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
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

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


json_file = open('model_big.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("weights_big.h5")

img_size = 224
img_size_touple = (img_size, img_size)

num_classes = 2
_images_per_file = 20

start = time.time()

class modelDone():

    def get_transfer_values(self, images):

        image_model = VGG16(include_top=True, weights='imagenet')

        transfer_layer = image_model.get_layer('fc2')
        image_model_transfer = Model(inputs=image_model.input,
                                    outputs=transfer_layer.output)
        transfer_values_size = K.int_shape(transfer_layer.output)[1]

        # Pre-allocate input-batch-array for images.
        shape = (_images_per_file,) + img_size_touple + (3,)

        image_batch = np.zeros(shape=shape, dtype=np.float16)

        image_batch = images

        # Pre-allocate output-array for transfer-values.
        # Note that we use 16-bit floating-points to save memory.
        shape = (_images_per_file, transfer_values_size)
        transfer_values = np.zeros(shape=shape, dtype=np.float16)

        print("vgg start: ", time.time() - start)
        transfer_values = image_model_transfer.predict(image_batch)
        print("vgg end: ", time.time() - start)

        return transfer_values

    def evaluation(self, images):

        image_set = self.get_transfer_values(images)
        image_set = np.reshape(image_set, [1,20,4096])

        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.predict(image_set)

        if score[0][0]>score[0][1]:
            return 1
        return 0




########### IGNORE THIS EXPERIMENTAL PART ###############
#     def run(self, file_name):

#         vidcap = cv2.VideoCapture(file_name)
#         flag = True

#         length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#         print( length )

#         answer = []
        
#         while flag:

#             print("loop start: ", time.time() - start)

#             images = np.zeros((20,224,224,3))
#             for i in range (0,20):

#                 success,image = vidcap.read()
#                 if success==False:
#                     flag = False
#                     break

#                 RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 img = cv2.resize(RGB_img, dsize=(img_size, img_size),interpolation=cv2.INTER_CUBIC)
#                 img = np.array(img)
#                 img = (img / 255.).astype(np.float16)
#                 images[i]=img

#             print("eval start: ", time.time() - start)
#             ans = self.evaluation(images) ## 20 , 224, 224, 3
#             print("ans = ",ans)
#             print("eval done: ", time.time() - start)

#             answer.append(ans)

            
#         return(answer)

# ob = modelDone()
# file_name = "test/2.avi"
# ans = ob.run(file_name=file_name)
# print(ans)


