This folder contains CNN for gender recognition based on face/facial features.
There are different python files:

  <b>faces_data.py</b> formats the data in the dataset folder and saves it as X.pickle and y.pickle
  
  <b>faces_train.py/faces_train_multi.py</b> trains the model but the differences between the two are those that
    _multi.py has different variations of models which you specify in the header.
      
      Statistics about those trained models are saves in /logs and the models could be saved but you have
      to uncomment the model.save in those python files.
      There are already some trained models in /models folder.
  
  <b>predict_in_input_photos.py</b> does preditions of faces in input /photos_file (faces have to be precropped-only face)
    prediction 0 = male
    preditction 1 = female
  
   <b>predict_in_video_create_video.py</b> predicts in real time gender of the people in the video for SPECIFIED number of frames
    which can be modified in the code. This video is also saved as <b>outpy</b> videofile.
    This predictor uses opencv haarcascade to extract faces from video and later crop/resize/convert to gray and use
    as a input for cnn.
    
https://github.com/PeterKillerio/Neural_Networks/blob/master/Tensorflow/Gender%20recognition%20with%20faces/faces/elon_original.png?raw=true
   
https://github.com/PeterKillerio/Neural_Networks/blob/master/Tensorflow/Gender%20recognition%20with%20faces/faces/elon_recognized.png?raw=true


Dataset could be downloaded from https://susanqq.github.io/UTKFace/ 
but I deleted the faces of people with age <8 because I cant differentiate the gender of those faces myself.
So find in code the directory of those faces and rename them I had my own faces_wo_kids (<8) dataset.
Dataset is loaded and formated to have 50/50 % gender ballance but loads just 4000 pictures. This can be modified in code.
As well as height/width can be modified in code, I used 100x100.

Preview of the network on video:
https://www.youtube.com/watch?v=6htumlO3MgU
