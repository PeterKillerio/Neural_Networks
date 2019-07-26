import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import time

#CREATE ALL COMBINATIONS FOR MODEL TRAINING
DENSE_LAYERS = [1,2,3,4,5]
LAYER_SIZES = [8,16,32,64,128]
EPOCHS = [1,3,25,50,100,200,400]

#LOAD DATA
flowers = []
path = 'Dataset/iris.csv'
flowers = pd.read_csv(path, names=['sepal_l', 'sepal_w', 'petal_l','petal_w','species'])
all_data = []

#FORMAT DATA TO ARRAY
for i in range(len(flowers)):
    if(i != 0):
        temp_data = []
        temp_data.append(float(flowers['sepal_w'][i])/8.0)
        temp_data.append(float(flowers['petal_w'][i])/8.0)
        temp_data.append(float(flowers['sepal_l'][i])/8.0)
        temp_data.append(float(flowers['petal_l'][i])/8.0)
        temp_data.append(flowers['species'][i])

        all_data.append(temp_data)

#RANDOMIZE ORDER
random.shuffle(all_data)

#FORMAT DATA FOR TRAINING
X_train = []
y_train = []

for data in all_data:
    X_train.append(data[:4])
    y_train.append(data[4])

#FORMAT LAST LAYER TRAIN DATA TO BE 0-1
encoder = LabelEncoder()
y1 = encoder.fit_transform(y_train)
Y = pd.get_dummies(y1).values

#X_t, X_test, y_t, y_test = train_test_split(X_train,Y, test_size=0.2, random_state=0)
#### Model declaration

for dense_lay in DENSE_LAYERS:
    for layer_siz in LAYER_SIZES:
        for epochs_num in EPOCHS:

            #CREATE NAME FOR MODEL TO BE SAVED IN LOGS
            NAME = "{}-DENSE_num-{}-LAYER_siz-{}-EPOCHS_num-{}".format(dense_lay, layer_siz, epochs_num, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))

            #MODEL DECLARATION
            model = Sequential()

            model.add(Dense(layer_siz, input_shape = (4, ),activation='relu'))
            model.add(Activation('relu'))

            for i in range(dense_lay-1):
                model.add(Dense(layer_siz))
                model.add(Activation('relu'))


            model.add(Dense(3))
            model.add(Activation('softmax'))

            model.compile(loss="binary_crossentropy",
                        optimizer="adam",
                        metrics=['accuracy'])

            X_train = np.array(X_train)

            #TRAINING
            model.fit(X_train, Y, epochs = epochs_num, validation_split = 0.1, callbacks=[tensorboard])

#SAVE MODEL

#model.save('test.model')
#model_test = tf.keras.models.load_model("test.model")
