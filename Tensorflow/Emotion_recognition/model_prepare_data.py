import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import pandas as pd
import pickle

#Path to training images
PATH_TO_IMAGES = '***********/facial_emotions/images/Training'
#Path to texting images
PATH_TO_TEST_IMAGES = '***********/facial_emotions/images/PublicTest'

TRAINING_DATA = []
TEST_DATA = []
################TRAINING DATA############################
#########################################################

#Find the names of all the images in a path and change the name to the corresponding global path of the image
images_names = os.listdir(PATH_TO_IMAGES)
for image_name in images_names:
    index_to_img = images_names.index(image_name)
    images_names[index_to_img] = PATH_TO_IMAGES + "/" + images_names[index_to_img]

#Load all the images, get the emotion number in an image name, change the pixel values to 0.0 to 1.0
#and add these data to the training images
for path in images_names:
    temp_data = []

    extract_info = path.split('/')
    extract_info = extract_info[-1].split('_')

    emotion = extract_info[1][0]

    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    image = image / 255
    img_w_channels = image

    temp_data.append(emotion)
    temp_data.append(img_w_channels)
    TRAINING_DATA.append(temp_data)
    print(images_names.index(path))

random.shuffle(TRAINING_DATA)
################TESTING DATA#############################
#########################################################

#Find the names of all the images in a path and change the name to the corresponding global path of the image
images_names = os.listdir(PATH_TO_TEST_IMAGES)
for image_name in images_names:
    index_to_img = images_names.index(image_name)
    images_names[index_to_img] = PATH_TO_TEST_IMAGES + "/" + images_names[index_to_img]

#Load all the images, get the emotion number in an image name, change the pixel values to 0.0 to 1.0
#and add these data to the training images
for path in images_names:
    temp_data = []

    extract_info = path.split('/')
    extract_info = extract_info[-1].split('_')

    emotion = extract_info[1][0]

    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    image = image / 255
    img_w_channels = image

    temp_data.append(emotion)
    temp_data.append(img_w_channels)
    TEST_DATA.append(temp_data)
    print(images_names.index(path))

#Shuffle the test_data
random.shuffle(TEST_DATA)
#########################################################
#########################################################

#PREPARING TRAINING DATA FOR EXPORT
#divide train data a config y
x_train_export = []
y_train_export = []

#Divide training data to the images(data[1]) and emotional labels (date[0])
for data in TRAINING_DATA:
    x_train_export.append(data[1])
    y_train_export.append(data[0])

#Create the last layer output format for model from the y_train_export (f.e. I have index 5 the code will create an array of size 7 and
#on the fifth index there will be a one value [0,0,0,0,0,1,0] )
encoder = LabelEncoder()
y_train_export_config = encoder.fit_transform(y_train_export)
y_train_export_config_final = pd.get_dummies(y_train_export_config).values

Y_train = np.array(y_train_export_config_final)
#divide test data and config y

x_test_export = []
y_test_export = []


for data in TEST_DATA:
    x_test_export.append(data[1])
    y_test_export.append(data[0])

#Create the last layer output format for model from the y_train_export (f.e. I have index 5 the code will create an array of size 7 and
#on the fifth index there will be a one value [0,0,0,0,0,1,0] )
encoder = LabelEncoder()
y_test_export_config = encoder.fit_transform(y_test_export)
y_test_export_config_final = pd.get_dummies(y_test_export_config).values

Y_test = np.array(y_test_export_config_final)

#EXPORT DATA
#x_train_export = np.array(x_train_export).reshape(-1, 48, 48, 1)
#x_test_export = np.array(x_test_export).reshape(-1, 48, 48, 1)

pickle_out = open("myown_X_train.pickle", "wb")
pickle.dump(x_train_export, pickle_out)
pickle_out.close()

pickle_out = open("myown_y_train.pickle","wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()
#export testing data
pickle_out = open("myown_X_test.pickle", "wb")
pickle.dump(x_test_export, pickle_out)
pickle_out.close()

pickle_out = open("myown_y_test.pickle","wb")
pickle.dump(Y_test, pickle_out)
pickle_out.close()
