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

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("final_weight.h5")

img_size = 224
img_size_touple = (img_size, img_size)
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 2
_num_files_train = 1
_images_per_file = 20
_num_images_train = _num_files_train * _images_per_file
video_exts = ".avi"


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

        transfer_values = \
                image_model_transfer.predict(image_batch)

        return transfer_values

    def evaluation(self, images):

        image_set = self.get_transfer_values(images)
        image_set = np.reshape(image_set, [1,20,4096])

        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.predict(image_set)

        if score[0][0]>score[0][1]:
            ans = 1
        else:
            ans = 0
        
        return ans
    


