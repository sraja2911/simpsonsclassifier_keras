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
import pickle 
from pickle import load
import pandas

#data_root = os.sep+os.path.join('tmp', 'simpsons')
data_root = '/data/trainingdata'
#dest_filename = os.path.join(data_root, 'simpsons_dataset.tar.gz')

#filename = '/home/raj/simpsonsZipData/simpsons_dataset.tar.gz'

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
  #print(data_folders)
  return data_folders
#data_folders = maybe_extract(dest_filename)
#data_folders = maybe_extract(filename)


img_width, img_height = 64, 64
#train_data_dir = os.sep+os.path.join('home', 'dagutman', 'devel', 'KerasSimpsons',  'simpsons_dataset')
train_data_dir = '/data/trainingdata/training' 
validation_data_dir = '/data/trainingdata/validation' 

nb_train_samples = 19000
nb_validation_samples = 890
epochs = 3
batch_size = 32


# Model definition
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


NumLabels = 18
    
'''
4-Conv layers
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NumLabels))
model.add(Activation('softmax'))

'''

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
    validation_data_dir + "/",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

simpsonsModel = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#model.save_weights('/data/trainingdata/simpsons_weights1.h5')
model.save('/data/trainingdata/simpsons_weights6.h5')

#pandas.DataFrame(simpsonsModel.history).to_csv("/data/trainingdata/simpsons_history.csv")
pandas.DataFrame(simpsonsModel.history).to_json("/data/trainingdata/simpsons_history.json")
                                         
#filename = open('/data/trainingdata/keras_output.txt', "a+")
#print >>filename, (simpsonsModel.history.keys())
#print >>filename, (simpsonsModel.history['acc'])
#print >>filename, (simpsonsModel.history['val_acc'])
#print >>filename, (simpsonsModel.history)
#print((simpsonsModel.history.keys(), file=filename))
#filename.close()

#with open('/data/trainingdata/trainHistoryDict', 'wb') as file_pi:
# pickle.dump(simpsonsModel.history, file_pi)

# summarize history for accuracy
# plt.plot(simpsonsModel.history['acc'])
# plt.plot(simpsonsModel.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# summarize history for loss
# plt.plot(simpsonsModel.history['loss'])
# plt.plot(simpsonsModel.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

