from tensorflow import keras
import tensorflow as tf
import pickle
import numpy as np
import face_recognition
import csv
import cv2
import os

#Load video
FILE_NAME = 'jim_carrey.mp4'
cap = cv2.VideoCapture(FILE_NAME)
#Those are emotions which you can recognize (later use their index)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#Load all the known faces and create encodings which will be used for recognition
DIRECTORY_PATH = '*************/facial_emotions'
FACES_PATH = "known_faces/"
face_encodings = []
face_names = []
known_faces = os.listdir(FACES_PATH)

for face in known_faces:
    face_image = face_recognition.load_image_file(DIRECTORY_PATH + "/" +FACES_PATH  + face)
    face_encoding = face_recognition.face_encodings(face_image)[0]
    name = face.split('.')[0]
    face_names.append(name)
    face_encodings.append(face_encoding)

#Load emotion recognition model
top_model = tf.keras.models.load_model('**************/facial_emotions/new_model.model')
#Create DATA_PACKETS array which will be exported after finishing all the recognizing
DATA_PACKETS = []
#This is the first line in a data file
DATA_PACKETS.append(('Frame','Name','Coordinates','Emotion',FILE_NAME))

########################
#Start of the main loop#
########################

#The number of frames you want to read from a video, after those frames the video will finish and the data will be saved
FRAMES_TO_READ = 250

#Frame number which symbolizes the frame sequence and is incremented after each loop
frame_number = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    frame_number += 1

    #frame = cv2.resize(frame, (400, 300))

    #Create RGB from BGR
    rgb_frame = frame[:, :, ::-1]

    #Detect all the faces and then create their encodings
    face_frame_locations = face_recognition.face_locations(rgb_frame)
    face_frame_encodings = face_recognition.face_encodings(rgb_frame, face_frame_locations)

    #Compare all the found faces with our known ones
    encoding_index = 0
    for face_frame_encoding in face_frame_encodings:


            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(face_encodings, face_frame_encoding)
            # If a match was found in known_face_encodings, just use the first one.
            print(face_frame_locations)
            if True in matches:
                first_match_index = matches.index(True)
                name = face_names[first_match_index]

                print("There is " + name)

                #Find an index of face_locations for compared face

                index_of_face_loc = encoding_index


                #Prepare the data for emotion recognition model
                y = face_frame_locations[index_of_face_loc][0]
                y_h = face_frame_locations[index_of_face_loc][2]
                x = face_frame_locations[index_of_face_loc][3]
                x_w = face_frame_locations[index_of_face_loc][1]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[y:y_h, x:x_w]

                img_predict = cv2.resize(roi_gray, (48, 48))
                img_predict = np.array(img_predict).reshape(-1, 48, 48, 1)

                #Recognize emotions
                answer = top_model.predict(img_predict)

                emotion_index = np.argmax(answer)
                emotion = emotions[emotion_index]

                #This array will be appended to the DATA_PACKETS with the desired information
                data_format = []
                data_format.append(frame_number)
                data_format.append(name)
                data_format.append(face_frame_locations[index_of_face_loc])
                data_format.append(emotion_index)

                DATA_PACKETS.append(data_format)

                text = emotion + " " + str(answer[0][emotion_index])
                print(text)

                print(data_format)

            encoding_index += 1

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    #If we reach the desired number of frames, end the loop
    if (frame_number >= FRAMES_TO_READ):
        #Write DATA_PACKETS to the csv file
        with open('video_data_rec.csv', 'w', newline='') as file:
            write_data = csv.writer(file)
            write_data.writerows(DATA_PACKETS)
            file.close()

        break
