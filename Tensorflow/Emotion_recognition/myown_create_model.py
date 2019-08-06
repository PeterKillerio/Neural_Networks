import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, InputLayer, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
import pickle
import numpy as np
import time

#Specify the model #You have to change it in the code as well
NUM_CLASSES = 7
OPTIMIZER = "adam"
CONV_NEURONS = 256
DENSE_NEURONS = 64
EPOCHS = 10
BATCH_SIZE = 25
DROPOUT = 0.2
NUMBER_OF_CONV = 4

#LOAD ALL NECESSARY DATA
pickle_in = open("myown_X_train.pickle", "rb")
X_train_f = pickle.load(pickle_in)

pickle_in = open("myown_X_test.pickle", "rb")
X_test_f = pickle.load(pickle_in)

pickle_in = open("myown_y_train.pickle", "rb")
Y_train = pickle.load(pickle_in)

pickle_in = open("myown_y_test.pickle", "rb")
Y_test = pickle.load(pickle_in)

#Format the data
X_train_f = np.array(X_train_f).reshape(-1, 48, 48, 1)
X_test_f = np.array(X_test_f).reshape(-1, 48, 48, 1)

print("********************")
print(X_train_f.shape)
print("********************")

#Create the model
Model = Sequential()

Model.add(Conv2D(256, (3,3), data_format = "channels_last",input_shape = (48,48,1)))
Model.add(Dropout(DROPOUT))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))


Model.add(Conv2D(256, (3,3)))
Model.add(Dropout(DROPOUT))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))

Model.add(Conv2D(256, (3,3)))
Model.add(Dropout(DROPOUT))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))

Model.add(Conv2D(256, (3,3)))
Model.add(Dropout(DROPOUT))
Model.add(Activation("relu"))


Model.add(Flatten()) #Converts 3D feature maps to 1D feature vectors

Model.add(Dense(DENSE_NEURONS))
Model.add(Dropout(DROPOUT))
Model.add(Activation('relu'))

Model.add(Dense(DENSE_NEURONS))
Model.add(Dropout(DROPOUT))
Model.add(Activation('relu'))

Model.add(Dense(7))
Model.add(Activation('sigmoid'))

#Declare the name of the save model
NAME_f = f"Conv_L_{NUMBER_OF_CONV}_Conv_{CONV_NEURONS}_Dense_{DENSE_NEURONS}_Epochs_{EPOCHS}_Batch_{BATCH_SIZE}_Dropout_{DROPOUT}"
NAME = NAME_f + str(int(time.time()))
tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))

Model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER, metrics=['accuracy'])

#Start training process
Model.fit(X_train_f, Y_train,
          validation_data=(X_test_f, Y_test),
          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tensorboard])

#Save the model
Model.save(NAME)
