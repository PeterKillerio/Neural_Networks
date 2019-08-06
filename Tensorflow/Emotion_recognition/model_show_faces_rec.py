import csv
import cv2

#Choose the index of the emotion you want to extract
#['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
SHOW_EMOTION = 3 #-2

#Declaring some global variables which will be used
#Video file name which will be automatically found in a csv file
FILE_NAME = ''
#Variable for saved images numbering
photo_number = 0
photos_data = []
#Array in which we will automatically store the names of the found peeople
NAME_LIST = []

#Varialble which is used for smoothing the face_recognition
#it checks how many times the emotion repeated and then it is used
EMOTION_REPEATER = 0

#Read the csv file, extract file name and get the data
with open('video_data_rec.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    lines = list(reader)
    FILE_NAME = str(lines[0][4])
    photos_data = lines[1:]

#Initializind the video file feed
cap = cv2.VideoCapture(FILE_NAME)

#Index which will be used as a way for us to know on which frame we are currenly at
temp_idx = 0

#Main loop
while(cap.isOpened()):
    ret, frame = cap.read()
    temp_idx += 1

    #If we cross the last frame in the data file, exit the program
    if(temp_idx >= (int(photos_data[-1][0]) - 1)):
        exit()

    #Only continue if there is at least some information in the line from the csv file
    if(len(photos_data[temp_idx]) > 1):

        #Extract the name of the person and append that person to our list
        name = photos_data[temp_idx][1]
        if(name not in NAME_LIST):
            NAME_LIST.append(name)

        #If the frame data matches our desired emotion, continue
        if(int(photos_data[temp_idx][3]) == SHOW_EMOTION ):

            #Check whether this emotion stays the same for declared number of frames
            till_end = int(photos_data[-1][0]) - temp_idx
            emotion_counter = 0
            print("Till end " + str(till_end))
            for i in range(till_end):
                print(i)

                if(int(photos_data[temp_idx + i][0]) <= (temp_idx + EMOTION_REPEATER)):
                    if(int(photos_data[temp_idx + i][3]) == SHOW_EMOTION):
                        emotion_counter = emotion_counter + 1
                else:
                    break

            print(emotion_counter)

            if(emotion_counter != EMOTION_REPEATER):
                continue


            #Get the face position data
            position_data = photos_data[temp_idx][2]
            print(position_data)

            position_data = position_data.replace("(", "")
            position_data = position_data.replace(")", "")
            position_data = position_data.replace(" ", "")

            position_data = position_data.split(',')
            #Get the correct format for face coordinates from face_recognition lib
            y = int(position_data[0])
            x_w = int(position_data[1])
            y_h = int(position_data[2])
            x = int(position_data[3])

            frame2 = frame[y:y_h, x:x_w]
            photo_number += 1

            cv2.imwrite("extracted_rec/" + name + "_"+str(photo_number)+'.jpg', frame2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
