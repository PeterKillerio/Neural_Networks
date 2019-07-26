import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

#Loading models
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#********************* UNCOMMENT AND CONFIG THESE LINES ********************

#Directory of your saved model
#model = tf.keras.models.load_model('C:/Users/******/faces/models/tensor_trained_model_ID2')

#Directory of your input photos they have to pre precropped
#DIRECTORY = "C:/Users/**********/faces/input_photos"

#***************************************************************************

PHOTO_SIZE = 100
files = os.listdir(DIRECTORY)
print(files)

for photo in files:
    pic = cv2.imread(os.path.join(DIRECTORY, photo), cv2.IMREAD_GRAYSCALE)
    pic = cv2.resize(pic,(PHOTO_SIZE, PHOTO_SIZE))
    cv2.imshow("hi", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    photo_data_array = []

    for row in pic:
        row_data = []

        for bits in row:
            row_data.append(bits/255.0)
        photo_data_array.append(row_data)

    ########## DATA READY

    X = []
    y = []


    for info in photo_data_array:
        X.append(info)


    X = np.array(X)
    X = np.expand_dims(X,axis=0)
    print(X.shape)
    X = np.swapaxes(X,0,1)
    X = np.swapaxes(X,1,2)
    X = np.expand_dims(X,axis=0)
    X = np.swapaxes(X,0,3)
    ######### DATA READY


    print(photo)
    prediction = model.predict(X)
    print(prediction)
    print("**************")

#prediction = model.predict(x_test)
