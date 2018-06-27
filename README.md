# Image Classifier for Simpsons Characters
A multi classes image classifier, based on convolutional neural network using Keras and Tensorflow. 
A multi-label classifier (having one fully-connected layer at the end), with multi-classification (18 classes, in this instance).
Largely copied from the code https://gist.github.com/seixaslipe
This is based on these posts: https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36
Data downloaded from Kaggle 

Will emulate the image classification functionlities for Neuro Pathology images/slides (WSI-Whole Slide images)
Will implement/include data manipulating functionalities based on Girder (https://girder.readthedocs.io/en/latest/)

Has 6 convulsions, filtering start with 64, 128, 256 with flattening to 1024
Used Keras.ImageDataGenerator for Training/Validation data augmentation and the augmented images are flown from respective directory
Environment: A docker container having Keras, TensorFlow, Python-2 with GPU based execution
