

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

