import csv
import cv2

#Declaring some global variables which will be used
#Video file name which will be automatically found in a csv file
FILE_NAME = ''
#Choose the index of the emotion you want to extract
#['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
SHOW_EMOTION = 7
#Variable for saved images numbering
photo_number = 0

photos_data = []

#Varialble which is used for smoothing the face_recognition
#it checks how many times the emotion repeated and then it is used
EMOTION_REPEATER = 8

#Read the csv file, extract file name and get the data
with open('video_data.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    lines = list(reader)
    FILE_NAME = str(lines[0][9])
    photos_data = lines[1:]

#Initialize video feed from file name which will be automatically read from csv file
cap = cv2.VideoCapture(FILE_NAME)

temp_idx = 0

#Start the main loop
while(cap.isOpened()):
    ret, frame = cap.read()
    temp_idx += 1

    #If we cross the last frame from the csv file, stop the video feed
    if(temp_idx >= int(photos_data[-1][0])):
        exit()

    #If there is one face in the frame start the algorithm
    if(int(photos_data[temp_idx][1]) == 1 ):
        #If we found the desired emotion
        if(int(photos_data[temp_idx][SHOW_EMOTION]) == 1 ):

            #EMOTION REPEATER CHECKING
            temp_data = 0
            for i in range(EMOTION_REPEATER):
                if (int(photos_data[temp_idx + i][SHOW_EMOTION]) == 1):
                    temp_data += 1
            if(temp_data != EMOTION_REPEATER):
                continue

            #Get the face coordinates of the person
            position_data = photos_data[temp_idx][9]
            #Format the coordinates data
            position_data = position_data.replace("(", "")
            position_data = position_data.replace(")", "")
            position_data = position_data.replace(" ", "")

            position_data = position_data.split(',')

            x = int(position_data[0])
            y = int(position_data[1])
            w = int(position_data[2])
            h = int(position_data[3])

            #Crop the face
            frame2 = frame[y:y+h, x:x+w]

            photo_number += 1
            #Save the face with corresponding photo_number numbering
            cv2.imwrite("extracted/"+str(photo_number)+'.jpg', frame2)

    #Show the video feed
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
