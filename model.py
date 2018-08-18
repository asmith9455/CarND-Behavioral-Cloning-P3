from data_exploration import explore_over_time, frame_count, generate_summary_plot

from file_contents_gen import get_batches

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D


fname = "../data/P3-sim-data"

print("starting linecount...")

line_count = frame_count(fname)

print("this data file contains ", line_count, " lines.")

line_count = 10

# //steering: -1 to 1
# // throttle 0 to 1
# // brake 0 1
# // speed 0 30

images, sw_angles, throttle, brake_input, speeds = explore_over_time(fname, line_count)

# generate_summary_plot(images, sw_angles, throttle, brake_input, speeds)

model = Sequential()

image_shape = images[0,0,:,:].shape

model.add(Lambda(lambda x: (x / 255.0 - 0.5) * 2, input_shape=image_shape))

model.add(Convolution2D(40, 5, 5, border_mode="valid"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Convolution2D(80, 5, 5, border_mode="valid"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Flatten())


model.add(Dense(400))

model.add(Activation("relu"))

model.add(Dense(200))

model.add(Activation("relu"))

model.add(Dense(1)) #steering wheel angle is the output

# features = images[:,0,:,:]
# labels = sw_angles

model.compile(loss='mse', optimizer='adam')

print('training start')

for batch_ctr, images, sw_angles, throttle, brake_input, speeds in get_batches(fname, 128):

    print('batch_ctr: ', batch_ctr)

    features = images[:,0,:,:]
    labels = sw_angles

    model.fit(features, labels, validation_split=0.3, shuffle=True, epochs=1, batch_size=64)

model.save('model.h5')
