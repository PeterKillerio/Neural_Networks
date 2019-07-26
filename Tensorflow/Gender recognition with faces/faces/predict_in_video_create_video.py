import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np



mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#********************* UNCOMMENT AND CONFIG THESE LINES ********************

#model = tf.keras.models.load_model('C:/Users/******************/faces/models/tensor_trained_model_ID1')

#Directory of your video
#cap = cv2.VideoCapture('video.mp4')

#***************************************************************************

PHOTO_SIZE = 100

#Output file of the captured video with anotations
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))

#Loading haarcascade for opencv to crop faces in video and the nuse the cropped faces
#to predict whether the face is of the male/female gender

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

index_photo = 0

#NUMBER OF PHOTOS TO SAVE TO A VIDEO FILE
DESIRED_INDEX = 250

while(cap.isOpened()):
    if (index_photo >= DESIRED_INDEX):
        break
    index_photo += 1

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        xx = x
        yy = y
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

        pic = cv2.resize(roi_gray,(PHOTO_SIZE, PHOTO_SIZE))
        photo_data_array = []

        for row in pic:
            row_data = []

            for bits in row:
                row_data.append(bits/255.0)
            photo_data_array.append(row_data)

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

        prediction = model.predict(X)
        print(prediction)

        print(xx)
        print(yy)
        if(prediction[0][0] > prediction[0][1]):
            cv2.putText(frame,'Male' + str(prediction[0][0]),(xx,yy),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0))
        else:
            cv2.putText(frame,'Female' + str(prediction[0][1]),(xx,yy),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0))


    cv2.imshow('frame',frame)

    out.write(frame)
    #break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
