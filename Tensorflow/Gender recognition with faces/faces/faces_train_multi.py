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

#Config for training
PHOTO_SIZE = 100
CONV_LAYERS = [3]
CONV_POOLS = [(2,2)]
CONV_SIZES = [60]
DENSE_LAYERS = [2]
DENSE_SIZES = [200]
EPOCHS = [6]
BATCH_SIZES = [20]
FILTER_SIZES = [(4,4)]

ALL_OPTIONS = len(CONV_LAYERS) *  len(CONV_POOLS) *  len(CONV_SIZES) *  len(DENSE_LAYERS) *  len(DENSE_SIZES) *  len(EPOCHS) *  len(BATCH_SIZES) *  len(FILTER_SIZES)


OPTIMIZER = "adam"
LOSS_f =  "binary_crossentropy"

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle", "rb")
Y = pickle.load(pickle_in)

#MODEL declaration and training
CL = 0
for CONV_LAYER in CONV_LAYERS:
    CP = 0
    for CONV_POOL in CONV_POOLS:
        CZ = 0
        for CONV_SIZE in CONV_SIZES:
            EP = 0
            for EPOCH in EPOCHS:
                BZ = 0
                for BATCH_SIZE in BATCH_SIZES:
                    FZ = 0
                    for FILTER_SIZE in FILTER_SIZES:
                        DZ = 0
                        for DENSE_SIZE in DENSE_SIZES:
                            DL = 0
                            for DENSE_LAYER in DENSE_LAYERS:
                                model = Sequential()

                                model.add(Conv2D(CONV_SIZE, FILTER_SIZE,input_shape=(PHOTO_SIZE, PHOTO_SIZE,1)))
                                model.add(Activation('relu'))
                                model.add(MaxPooling2D(pool_size=CONV_POOL))

                                for i in range(CONV_LAYER):
                                    model.add(Conv2D(CONV_SIZE, FILTER_SIZE))
                                    model.add(Activation('relu'))
                                    model.add(MaxPooling2D(pool_size=CONV_POOL))

                                model.add(Flatten()) #Converts 3D feature maps to 1D feature vectors

                                for ii in range(DENSE_LAYER):
                                    model.add(Dense(DENSE_SIZE))
                                    model.add(Activation('relu'))
                                    model.add(Dropout(0.1))

                                model.add(Dense(2))
                                model.add(Activation('softmax'))

                                NAME = f"{CONV_LAYER}-conv_howmany-{CONV_SIZE}-CONV_NUM_siz-{DENSE_LAYER}-dense_layerhowmany -{DENSE_SIZE}-dense_sizes-{EPOCHS}-EPOCHS_num-{BATCH_SIZE}-batch_num-{CONV_POOL}-CONV_POOLS-{FILTER_SIZE}-filter_size-{int(time.time())}"
                                tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))

                                model.compile(loss=LOSS_f,
                                            optimizer=OPTIMIZER,
                                            metrics=['accuracy'])
                                SUM_OF_ALL = CL + CP + CZ + EP + BZ + FZ + DZ + DL

                                print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
                                print(f"WE ARE AT: {SUM_OF_ALL}/{ALL_OPTIONS} ... {(SUM_OF_ALL/ALL_OPTIONS)*100}%")
                                print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

                                model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.1, callbacks=[tensorboard])

                                #Save your model
                                #model.save('tensor_trained_model')
