Transfer learning of Object Detection Mobilenet V1 COCO Model for Hand Detection

This was the first neural network model I could call my own in the sense that I created my own dataset which 
consisted of over 600 labeled instances of hands,
still I followed an amazing tutorial by Sentdex which can be found here:

https://www.youtube.com/watch?v=COlbP62-B-U&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku

and installation guide here:

https://github.com/tensorflow/models/tree/master/research/object_detection

Its tedious too start and get it going, I had to spend some hours of debugging and learning.
But the whole process took a lot more steps that I expected in order to do 
transfer learning on some existing model for object detection in my case MobileNet V1 COCO. But after hours and hours it magically worked. 

<b>Example video:</b> https://www.youtube.com/watch?v=2LuNuRDRcN8

![alt text](https://github.com/PeterKillerio/Neural_Networks/blob/master/Tensorflow/Hand_detection/elon_original.jpg)
![alt text](https://github.com/PeterKillerio/Neural_Networks/blob/master/Tensorflow/Hand_detection/elon_recognized.png)
![alt text](https://github.com/PeterKillerio/Neural_Networks/blob/master/Tensorflow/Hand_detection/trump_original.jpg)
![alt text](https://github.com/PeterKillerio/Neural_Networks/blob/master/Tensorflow/Hand_detection/trump_recognized.png)
![alt text](https://github.com/PeterKillerio/Neural_Networks/blob/master/Tensorflow/Hand_detection/google_search.png)

I could detect hand in an image and video in real time just on CPU. 
FPS weren’t very high but these things are recommended to run on GPU/TPU, but you could optimize for speed by converting 
the model to tflite and by using quantized model mobilenet as well as using post training quantization 
(I don’t have experience with these techniques, just a rather theoretical understanding but I plan 
to use my models on android so I will try to work tflite models).

This model I did isn’t very robust and there could be couple of reasons for that:
Hand is rather more complicated thing to detect if you take into account all the shapes and all the angles 
from which the hand can be observed. 

In my training data there were various cases of hands but after 
I retrained the model I theorized if my success rate would improve by not adding hand in pockets 
and hands beside body into my training dataset. If I was f.e. using my model to detect waving I would include 
just the images of hand facing to and from the camera view. 

Nice project for this would be the possibility 
to play rock, paper, scissors with your computer in that sense you would have to think about the angle of the camera 
and try to find the dataset dedicated to this. Google had a conference about TensorFlow library 
few months back at the time of writing this article and the example with which they presented the power of this 
library was specifically the rock, paper, scissor game. The difference was that they didn’t create any 
object detection or were playing the came from more difficult angles. They were playing the game from 
top view on the same surface and the hands were in the same position, I’m not sure about the robustness 
but it’s still a great and simple example.

Back to my model. 

Transfer learning using object detection model from TensorFlow is not a very 
creative task about which I could write about. You just need to set up everything, 
download datasets, download the model you want to retrain, label dataset 
(which could take you couple of hours of manual work but it’s worth it). 
Change a few lines of code in python files and then hit enter. So basically, 
you can do this with any object, but you can still specify some model characteristics.

	
