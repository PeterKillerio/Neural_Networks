import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


#********************* UNCOMMENT AND CONFIG THESE LINES ********************

#Your dataset directory
#DATADIR = "C:/Users/*******/faces/face_database_wo_kids"


#***************************************************************************

PHOTOS_DIR = [os.listdir(DATADIR)]

#Photo size both width and height
PHOTO_SIZE = 100

#Its the limit where I should stop loading the pictures in order to maintain 50/50 ballance between genders.
GENDER_LIMIT = 4370 #3105 #4370

training_data = []

#Limit index for loading only 4000 pictures 
#IDX HAS TO BE LOWER THAN GENDER LIMIT ! 
IDX = 4367

def create_empty_array(length):
    a = []
    for i in range(length):
        a.append(0)
    return a

def create_training_data():
    IDX_NOW = 0
    photos = os.listdir(DATADIR)

    array = [0,0]

    for photo in photos:
        photo_data_array = []

        img = cv2.imread(os.path.join(DATADIR, photo), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(PHOTO_SIZE,PHOTO_SIZE))

        info = photo.split("_")

        #DATA 50% TO 50% GENDER
        if (int(info[1]) == 0):
            if(array[0] > GENDER_LIMIT):
                continue
            array[0] += 1
        elif(int(info[1]) == 1):
            if(array[1] > GENDER_LIMIT):
                continue
            array[1] += 1


        for row in img:
            row_data = []

            for bits in row:
                row_data.append(bits/255.0)
            photo_data_array.append(row_data)

        photo_data_array.append(info[1])
        training_data.append(photo_data_array)

        random.shuffle(training_data)

        IDX_NOW += 1

        if (IDX <= IDX_NOW):
            break

    print(array[0])
    print(array[1])

create_training_data()

X = []
y = []

for info in training_data:
    X.append(info[:(len(info)-1)])
    y.append(info[-1])

X = np.array(X)
print(X.shape)
X = np.swapaxes(X,0,1)
X = np.swapaxes(X,1,2)
X = np.expand_dims(X,axis=0)
X = np.swapaxes(X,0,3)
print(X.shape)

#Prepare lastlayer training data for back propagation 0-1
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values

Y = np.array(Y)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
