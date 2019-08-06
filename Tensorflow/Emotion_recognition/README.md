<b>Example video:</b> https://www.youtube.com/watch?v=PdgOubpjWac

I think this was my ninth neural network per se that I’ve build and tested. 
And fourth in the meaning that the objective of the neural network was somewhat “original” in the sense that I didn’t follow any full sized step by step tutorial but relied on previously build neural networks, online information about the topic, recommendations as well as finding online datasets and one time creating my own dataset for hand detection. 

I am using high level programming open source library keras with TensorFlow which 
is a rather “easy” to start tool for creating your own neural networks. For example, 
I don’t exactly understand different back propagation methods as of yet, some specific 
mathematical equations behind the scene but right now I don’t need to. I have general 
representation of the function of the network and its parts. With this I think 
I would be able to create my own from scratch as I tried with book called 
(Make Your Own Neural Network by Tariq Rashid, which introduces you to the 
low level programming of neural network but in the most basic example) but I would 
have to study every single mathematical expression used and would have to really put 
elbow grease and countless hours into it.

I am specifically talking just about CNNs (convolutional neural networks), which are 
mostly used for image recognition/detection and classical dense layers NNs. But I also 
did NN with LSTM cells but that was tutorial example.

As they say the repetition is the mother of learning, I wanted to make more neural 
networks whose objective I chose myself to get the creation/prototyping of CNNs into my muscle memory.

I was thinking what would be interesting CNN to create/show/practice and build upon 
and I thought of a little project whose objective was to extract faces of people from 
video/movie based upon the emotion and optionally a specific persona. 

I divided this project into specific parts which were. 

<b>
1.	Read video/camera feed and get images

2.	Get faces from these images

3.	Get only images which are aligned to be recognized in other words recognize if there are eyes in the face picture

4.	Crop this face picture from image

5.	Feed this face picture into the face emotion recognition CNN

a.	Get the dataset

b.	Prepare the dataset

c.	Create model for CNN

d.	Train, test, train, test, train, test, train…….

e.	Deploy

6.	Get the emotions and save them accordingly to some data format which will be then exported.

7.	Export data to the csv file

8.	Import the data from csv file and apply it on video to crop/extract face with desired emotion</b>


As I worked with OpenCV Python library before I knew what it is capable of and it was my first option.
I specifically used OpenCV mainly for points 1 to 4 and 8. 
TensorFlow library was with combination used for point 5 and then I used csv library for exporting.

I don’t have intention to make this article a full-sized tutorial with code examples but rather general talk on how I did it.

<b>Point 1</b>
Getting the video/camera feed and getting the images is very simple task for which I used OpenCV python library. This can really be done with couple lines of code. There are numerous examples on how to do this both in Python as well as in C++ and I would recommend for starters to fly through some general OpenCV tutorial like on pythonprogramming.net/tutorial on YouTube from sentdex.

<b>Point 2</b>
My intention wasn’t to reinvent the wheel in face detection so I used haarcascades which are open source xml files/classifiers which can be imported to use with OpenCV. As far as I know they have license from Intel and can’t be used commercially. If you want to use it commercially, I would check out the face_recognition python library which has deep NN for face detection implemented and is open-source or use any other way you can find.

<b>Point 3</b>
As I mentioned I wanted to use faces which are aligned i.e. they have frontal face in the picture I thought that nice way to accomplish this would be to have eye detection within face detection so the face is acknowledged only when there are 2+ eyes in the picture. 2+ because sometimes the OpenCV recognizes eyes in mouth corners etc. In order to use eye detection, you can download yet another haarcasecade xml file. Just search haarcascades OpenCV Python GitHub and you will find couple of the and there should be at least one with the name eye detection.

<b>Point 4</b>
To be able to crop face from the picture you need the data about the position of the left upper corner and right down corner. This data can be saved from the previous use of face detection which haarcascade detection command returns x,y,w,h (x, y, width, height) coordinates. You just the need to use this data as arguments in OpenCV command for cropping picture, again just quick search will get you to the command.

<b>Point 5</b>

<b>Transfer learning</b>
For me in order to detect emotions in pictures I need to have emotion detector implemented. 
So, I used deep CNNs because I don’t know about any other way how it could be done, maybe 
hardcoding features and filters but that would be a mess. 
My first thought was that there are already implemented models for object recognition i.e. 
models which were trained on thousands/millions of images to extract information/features from them. 
So I decided to use Transfer learning, which means that I am going to use already trained neural network 
but I will throw out the dense/top layers at the end of the network which are basically 
just using the information it has about the features and assigns it to specific item/object like car, face. 
If I get rid of those layers, I will be able to get the features from the image I pass in. 
But in order to distinguish those features I will need yet another neural network this time dense layers. 
So, my workflow was this use some existing CNN model (I used vgg16), delete top 2 layers, 
test it by passing picture (which in my case of the vgg16 the 224x224x3 was the required format) 
and storing and checking the output.
The dataset I used was from Kaggle site and it was once used for competition. You can find it out by 
typing (Kaggle Challenges in Representation Learning: Facial Expression Recognition Challenge) into the search bar.
Once I had the dataset, I used script to convert the csv dataset file into images, 
this script can be found here:  https://github.com/abhijeet3922/FaceEmotion_ID (csv_to_images.py)
Then I just loaded the images using OpenCV and then rescale them to the desired CNN input, 
added 2 more channels because the pictures were black/white and the input required 3 channels.
This data was shuffled and then exported using python library pickle into my directory.
I fed the network the pickle data, got the output, and using pickle I exported the the output of the images.  
*I also could have checked if the proportion of the emotions was the same i.e. that there were the same number of happy,sad… faces.
And change the proportions appropriately.*
Then I used this exported data (image data, labels) as an input to my newly created model which consisted of 
3 dense layers and trained the model. I could’t get over 44% and when I tested the model real time 
using my webcam it was horrible. After checking the output, I discovered that the output from the VGG16 
was very binary-like, most of the values were zero and the ouputs from the implemented model were jumping 
like crazy so I tried another strategy.

<b>CNN implementation</b>
After this unsuccessful attempt I just decided to create my own CNN, using the code I used for creating past
CNNs I added few layers, changed the input, change the data format (I no longer needed 3 channels) and started training. 
I trained at least 20+ models and watched how the validation loss/success rate varied and settled on solid 57% from 
7 different expressions. After I found out that one epoch would take 17 minutes to train on my CPU I decided to use 
cloud gpu service called paperspace which I also used on hand detection (paperspace let you do your training on 
virtual linux, ubuntu desktop, all you need to do is to choose ML in a box option). I you want 10$ credit you started 
just use my referral link: https://paperspace.io/&R=BE4CKHI. The cloud gpu training compared to my cpu was about 30-40x faster option.
After I tested my models, I decided that it was good enough and as I mentioned I settled on 57%. 
To improve this model, I also moved over 5000 pictures from testing dataset to training dataset and my
training dataset had 1000 pictures and my training over 33 000 I think.

<b>Point 6</b>
All I had to do right now was just to tinker. Use this model automatically to detect emotions on every frame 
of the video/camera, create rectangles for eyes, face and add text of emotion next to every face.

<b>Point 7-8</b>
After that I created data format which will be used to store my data about the video. I used data format like this.
[frame, face, 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
First two columns told on which frame the data had been recorded and how many faces there are,
then every other column was zero and after an emotion has been detected the emotion incremented.
The first row also had filename at the end to automate the process.
After face has been detected I also added another rows with face coordinates.

This was a rather dumb way to extract emotions because you don’t know which face had which emotion 
and it wouldn’t be as nice to do something like each face has emotion because not each face had emotion i.e. 
they had to have eyes and they didn’t have eyes every time. Simply I settled on this solution but there is room for improvement.
If I wanted to extract only one face with emotion I had to check if there is only one face in the picture 
and then check the desired index. Then export the picture into the folder and look at them.

My CNN model is far from perfect and twitches a lot, sometimes neutral face is happy face and it is for 
this network also hard to distinguish emotions when people talk. Try pausing video of someone talking and 
tell me their emotion, it’s sometimes hard for people as well.

So, I decided to implement a new feature to the algorithm, I called it emotion_repeater and it 
checks how many times in a row emotion didn’t change. I saw that prediction twitched like hell when 
the person was speaking but calmed down when he/she didn’t. The emotions relatively stayed the same for more frames, 
they were more consistent. After changing the values from 2 to 5 to 10 I saw very noticeable increase 
and I was really happy with the result.

<b>Without repeater</b>
![alt text]( https://github.com/PeterKillerio/Neural_Networks/blob/master/Tensorflow/Emotion_recognition/happy_faces_1.png)
<b>With repeater</b>
![alt text]( https://github.com/PeterKillerio/Neural_Networks/blob/master/Tensorflow/Emotion_recognition/happy_faces_repeater.png)

So you can try it yourself, just add a video to my code directory, 
change the name of the file in the generate data python file, generate data, 
start extracting emotion (choose emotion in the header), adjust repeater and hit start. 

<b>Extraction example:</b> 
https://www.youtube.com/watch?v=z3NywmE1fjo


