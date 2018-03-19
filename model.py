import os
import csv
import cv2
import numpy as np
import sklearn

from pathlib import PureWindowsPath
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

# for one line of csv, we are adding 6 images to the samples
MULTI_FACTOR = 6
BATCH_SIZE = 64

def get_data_from_image(name, measurement, correction, images, angles):
    """Get the images and angles data for normal and flipped image
    Adds image and steering angle to the images and angles list inline
    Flips the image vertically and adds to image and steering angle is multiplied by -1 and then added"""
    name = name.strip()
    if name:
        try:
            image = cv2.imread(name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print('error while opening image file: {}, measurement: {}'.format(name, measurement))
            return
        angle = measurement + correction
        images.append(image)
        angles.append(angle)
        # if abs(angle) >= 0.2:
            # flip vertically only if the steering angle is more than .2
        images.append(cv2.flip(image, 1))
        angles.append(-angle)

def generator(samples, batch_size=BATCH_SIZE):
    """Generator for the image"""
    num_samples = len(samples)
    multi_f = 0.25
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    if i % 3 == 0: #center image
                        correction = 0
                    elif i % 3 == 1: # left image
                        correction =  multi_f
                    else: # right image
                        correction = -multi_f
                    # print('i: ', i, 'correction', correction)
                    # below method adds the images and angles data inline to the images and angles list
                    get_data_from_image(batch_sample[i], float(batch_sample[3]), correction, images, angles)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)  
    
def get_samples():
    """Get the samples from the csv file
    """
    samples = []
    #number of directories which contains the data
    data_dir = 'data'
    data_count = len(next(os.walk(data_dir))[1])
    # data_count = 2
    print ('total data folders are:', data_count)

    for i in range(data_count):
        csv_file_path = os.path.join(data_dir, str(i), 'driving_log.csv')
        with open(csv_file_path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip the headers
            for line in reader:
                for x in range(3):
                    file_path = PureWindowsPath(line[x]).parts[-1]
                    line[x] = os.path.join('.', data_dir, str(i), 'IMG', file_path)
                samples.append(line)
    return samples

def get_model():
    """Get the model"""
    input_shape = (160, 320, 3)
    model = Sequential()
    
    # create nVidia model
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(input_shape=input_shape, cropping=((58,24), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
    
train_samples, validation_samples = train_test_split(get_samples(), test_size=0.2)

TOTAL_TRAINING_SAMPLE_COUNT = len(train_samples) * MULTI_FACTOR
TOTAL_VALID_SAMPLE_COUNT = len(validation_samples) * MULTI_FACTOR

print('train sample count: ', TOTAL_TRAINING_SAMPLE_COUNT)
print('validation sample count: ', TOTAL_VALID_SAMPLE_COUNT)

# get generators
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = get_model()
model.summary()
# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png', show_shapes=True)

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, 
                              samples_per_epoch=TOTAL_TRAINING_SAMPLE_COUNT, 
                              validation_data=validation_generator,
                              nb_val_samples=TOTAL_VALID_SAMPLE_COUNT, 
                              nb_epoch=3)
                              
model.save('neeraj_model.h5')

print(history.history.keys())
print('Loss: ', history.history['loss'])
print('Validation Loss: ', history.history['val_loss'])