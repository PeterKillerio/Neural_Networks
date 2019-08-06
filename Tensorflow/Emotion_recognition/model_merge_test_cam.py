from tensorflow import keras
import tensorflow as tf
import pickle
import numpy as np
import cv2

#Load the recognition model
top_model = tf.keras.models.load_model('**********/facial_emotions/new_model.model')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#Start capturing the video (webcam)
cap = cv2.VideoCapture(0)
#Import face detection haarcascade to opencv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

xx = 0
yy = 0
ww = 0
hh = 0

while(True):
    #Read cam image
    ret, frame = cap.read()


    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop_img = img

    #Detect faces
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    #For each detected face draw rectangles and crop one face from the webcam on which we will run emotion detection
    for (x,y,w,h) in faces:
        xx = x
        yy = y
        hh = h
        ww = w
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        crop_img = img[y:y+h, x:x+w]

    #Prepare the data for input to emotion detection model
    img_predict = cv2.resize(crop_img, (48, 48))
    img_predict = np.array(img_predict).reshape(-1, 48, 48, 1)

    ################################
    ################################

    #Predic the emotion from cropped face
    answer = top_model.predict(img_predict)
    print(emotions[np.argmax(answer)])

    #Draw text to the face rectangle
    cv2.putText(frame,emotions[np.argmax(answer)],(xx+ww,yy+hh), 2, 1,(0,0,0),1,cv2.LINE_AA)
    ################################

    #Show images
    cv2.imshow('image',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
