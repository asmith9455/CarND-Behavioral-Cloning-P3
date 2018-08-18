from data_exploration import explore_over_time, frame_count, generate_summary_plot
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D


fname = "../data/P3-sim-data"

line_count = frame_count(fname)

line_count = 4000


print("this data file contains ", line_count, " lines.")

# //steering: -1 to 1
# // throttle 0 to 1
# // brake 0 1
# // speed 0 30

images, sw_angles, throttle, brake_input, speeds = explore_over_time(fname, line_count)

generate_summary_plot(images, sw_angles, throttle, brake_input, speeds)


model = Sequential()

image_shape = images[0,0,:,:].shape

model.add(Lambda(lambda x: (x / 255.0 - 0.5) * 2, input_shape=image_shape))

model.add(Convolution2D(10, 5, 5, border_mode="valid"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))




model.add(Convolution2D(20, 5, 5, border_mode="valid"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Flatten())


model.add(Dense(250))

model.add(Activation("relu"))

model.add(Dense(500))

model.add(Activation("relu"))

model.add(Dense(200))

model.add(Activation("relu"))

model.add(Dense(1)) #steering wheel angle is the output

features = images[:,0,:,:]
labels = sw_angles

model.compile(loss='mse', optimizer='adam')
model.fit(features, labels, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('model.h5')