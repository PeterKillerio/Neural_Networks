This folder contains CNN for gender recognition based on face/facial features.
There are different python files:

  faces_data.py formats data in the dataset folder and saves it as X.picle and y.picle
  faces_train.py/faces_train_multi.py trains the model but the differences between the two are those that
    _multi.py has different variations of models which you specify in the header.
      
      Statistics about those trained models are saves in /logs and the models could be saved but you have
      to uncomment the model.save in those python files.
      There are already some trained models in /models folder.
  
  predict_in_input_photos.py does preditions of faces in input /photos_file (faces have to be precropped)
    prediction 0 = male
    preditction 1 = female
  
  predict_in_video_create_video.py predict in real time gender of the people in the video for SPECIFIED number of images
    which can be modifie in the code. This video is also saved as outpy videofile.
    This predictor uses opencv haarcascade to extract faces from video and later crop/resize/convert to gray and use
    as a input for cnn.
    
Dataset could be downloaded from https://susanqq.github.io/UTKFace/ 
but I deleted the faces of people with age <8 because I cant differentiate the age of those faces myself.
So find in code the directory of those faces and rename them I had my own faces_wo_kids (<8) dataset.
Dataset is loaded and formated to have 50/50 % gender ballance but loads just 4000 pictures. This can be modified in code.
As well as height/width can be modified in code, I used 100x100.
