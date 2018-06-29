# A multi classes image classifier, based on convolutional neural network using Keras and Tensorflow. 
# A multi-label classifier (having one fully-connected layer at the end), with multi-classification (18 classes, in this instance).
# Largely copied from the code https://gist.github.com/seixaslipe
# This is based on these posts: https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36
# Data downloaded from Kaggle 
# Will emulate the image classification functionlities for Neuro Pathology images/slides (WSI-Whole Slide images)
# Will implement/include data manipulating functionalities based on Girder (https://girder.readthedocs.io/en/latest/)
# Has 6 convulsions, filtering start with 64, 128, 256 with flattening to 1024
# Used Keras.ImageDataGenerator for Training/Validation data augmentation and the augmented images are flown from respective directory
# Environment: A docker container having Keras, TensorFlow, Python-2 with GPU based execution

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
import sys
import tarfile
import numpy as np
import h5py
import matplotlib as plt
plt.use('Agg')
import matplotlib.pyplot as pyplot
pyplot.figure
import pickle 
from pickle import load
import pandas
from sklearn.metrics import classification_report, confusion_matrix

with open("epochout.txt", "w") as file:
    sys.stdout = sys.stderr = file 

data_root = '/data/trainingdata'
predict_data_dir = '/data/testimagedata'

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  return data_folders
#data_folders = maybe_extract(dest_filename)
#data_folders = maybe_extract(filename)


img_width, img_height = 64, 64
train_data_dir = '/data/trainingdata/training' 
validation_data_dir = '/data/trainingdata/validation' 

nb_train_samples = 19000
nb_validation_samples = 890
epochs = 2
batch_size = 32


# Model definition
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
NumLabels = 18
    
'''
6-conv layers - added on 06/21, Raj
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same')) 
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NumLabels, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip=True)

# Only rescaling for validation
valid_datagen = ImageDataGenerator(rescale=1. / 255.0)

# Flows the data directly from the directory structure, resizing where needed
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

simpsonsModel = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

print ("model generated")

model.save('/data/trainingdata/simpsons_weights6_0629.h5')

pandas.DataFrame(simpsonsModel.history).to_json("/data/trainingdata/simpsons_history_0629.json")

# printing Confusion Matrix and Classification Report
target_names = ['grampa',  'apu', 'bart',  'burns',  'chief', 'comic', 'edna', 'homer', 'brockman',
            'krusty',  'lisa', 'marge', 'milhouse', 'moe', 'ned', 'nelson', 'skinner', 'sideshow']

# saving Confusion Matrix and Classification Report to a file
file = open('simpsons_output.txt', "a+")
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
ptropt= 'Confusion Matrix' 
print >> file, ptropt
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)
print >>file, cnf_matrix
ptropt = 'Classification Report'
print >> file, ptropt
cls_rpt = classification_report(validation_generator.classes, y_pred, target_names=target_names) 
print >> file, cls_rpt
file.close()                                         


#Confusion Matrix is shown on a Plot
chardict = {'grampa': 0,  'apu': 1, 'bart': 2,  'burns': 3,  'chief': 4, 'comic': 5, 'edna': 6, 'homer': 7, 'brockman': 8,
            'krusty': 9,  'lisa': 10, 'marge': 11, 'milhouse': 12, 'moe': 13, 'ned': 14, 'nelson': 
             15, 'skinner': 16, 'sideshow': 17}
pyplot.figure(figsize=(8,8))
cnf_matrix =confusion_matrix(validation_generator.classes, y_pred)
classes = list(chardict.values())
pyplot.imshow(cnf_matrix, interpolation='nearest')
pyplot.colorbar()
tick_marks = np.arange(len(classes))  
_ = pyplot.xticks(tick_marks, classes, rotation=90)
_ = pyplot.yticks(tick_marks, classes)
pyplot.savefig('simpsons_predict_0629.png')


# To extract epoch run time
import re
pattern = re.compile(r's/step*')

epochs=[]
with open("epochout.txt") as f:
    for line in f:
        epochs=pattern.search(line).group():
            

