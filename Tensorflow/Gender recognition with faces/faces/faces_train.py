import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.models import Sequential
import time

PHOTO_SIZE = 100
OPTIMIZER = "adam"
LOSS_f =  "binary_crossentropy"

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle", "rb")
Y = pickle.load(pickle_in)

#MODEL declaration
model = Sequential()

model.add(Conv2D(50, (3,3),input_shape=(PHOTO_SIZE, PHOTO_SIZE,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(50, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #Converts 3D feature maps to 1D feature vectors

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

#NAME = "{}-DENSE_num-{}-LAYER_siz-{}-EPOCHS_num-{}".format(dense_lay, layer_siz, epochs_num, int(time.time()))
NAME = str(int(time.time()))
tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))

model.compile(loss=LOSS_f,
            optimizer=OPTIMIZER,
            metrics=['accuracy'])

model.fit(X, Y, batch_size=5, epochs=3, validation_split=0.1, callbacks=[tensorboard])
