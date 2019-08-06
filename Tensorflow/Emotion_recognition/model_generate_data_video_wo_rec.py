from tensorflow import keras
import tensorflow as tf
import pickle
import numpy as np
import face_recognition
import csv
import cv2

#Load video
FILE_NAME = 'jim_carrey.mp4'
cap = cv2.VideoCapture(FILE_NAME)

#Load face and eye detection clasiffiers into opencv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Emotion array
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#Data headers which will be saved in csv file
data_headers =  ['Frame','Faces','Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral',FILE_NAME]
data_packets = []
data_packets.append(data_headers)

#Total number of frames to be read
FRAMES_TO_READ = 2500

#Path and loading of the model
top_model = tf.keras.models.load_model('***************/facial_emotions/new_model.model')

frame_number = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    frame_number += 1

    #Initialize empy data_format
    data_format = [0,0,0,0,0,0,0,0,0]
    data_format[0] = frame_number

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cropped_faces = []

    #Face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)

    #For each detected face detect eyes and if there are
    #at least 2 eyes in the face, crop this face and draw rectangles in frame image
    for (x,y,w,h) in faces:
        #ADD data faces
        data_format[1] = len(faces)
        data_format.append((x,y,w,h))

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if (len(eyes) >= 2):
            img_predict = cv2.resize(roi_gray, (48, 48))
            img_predict = np.array(img_predict).reshape(-1, 48, 48, 1)
            cropped_faces.append((img_predict,(x+w,y+h)))

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(x + ex,y + ey),(x + ex+ew,y + ey+eh),(0,127,255),2)




    #For all those cropped faces, run emotion detection, add data information about the emotion and draw text on frame image
    for people_faces in cropped_faces:
        answer = top_model.predict(people_faces[0])

        emotion_index = np.argmax(answer)
        emotion = emotions[emotion_index]

        text = emotion + " " + str(answer[0][emotion_index])
        cv2.putText(frame,text,(people_faces[1][0],people_faces[1][1]), 2, 1,(0,0,0),1,cv2.LINE_AA)
        #ADD EMOTION DATA
        data_format[emotion_index+2] += 1



    #Display the frame images
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #Append a row of data to the data_packets(will be saved as a whole csv file)
    data_packets.append(data_format)

    #If we reach our desired frames read, save the video data file
    if (frame_number >= FRAMES_TO_READ):

        with open('video_data.csv', 'w', newline='') as file:
            write_data = csv.writer(file)
            write_data.writerows(data_packets)
            file.close()

        break
