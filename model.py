from data_exploration import explore_over_time, frame_count, generate_summary_plot

from file_contents_gen import get_batches_multi_dir

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Lambda, Dropout
# from tf.keras.layers import InputLayer
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from  import tf as ktf
# import tensorflow as tf
# import keras

import matplotlib.pyplot as plt
import numpy as np

import keras


fname = "../data/P3-sim-data-udacity/data"
# fname = "../data/P3-sim-data-5"

dirs = ["../data/P3-sim-data-udacity/data"]

# //steering: -1 to 1
# // throttle 0 to 1
# // brake 0 1
# // speed 0 30

images, sw_angles, throttle, brake_input, speeds = explore_over_time(fname, 300)

generate_summary_plot(images, sw_angles, throttle, brake_input, speeds)

model = Sequential()

image_shape = (70, 160, 3)# images[0,0,:,:].shape

# model.add(InputLayer(input_shape=(None, 160, 320, 3)))

model.add(Lambda(lambda x: __import__('tensorflow').image.rgb_to_grayscale(x)))

model.add(Cropping2D(cropping=( (60,0), (0,0) )))

model.add(Lambda(lambda x: __import__('keras').backend.tf.image.resize_images(x, (50,160))))

model.add(Lambda(lambda x: (x / 255.0 - 0.5) * 2))

model.add(Conv2D(filters=12, kernel_size=5, strides=(1,1), activation='relu'))

model.add(Conv2D(filters=24, kernel_size=5, strides=(2,2), activation='relu'))

model.add(Conv2D(filters=36, kernel_size=5, strides=(2,2), activation='relu'))

model.add(Conv2D(filters=48, kernel_size=3, strides=(1,1), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=3, strides=(1,1), activation='relu'))

model.add(Flatten())

model.add(Dense(400, activation='relu'))

model.add(Dense(600, activation='relu'))

model.add(Dense(300, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1)) #steering wheel angle is the output

# features = images[:,0,:,:]
# labels = sw_angles

opt = keras.optimizers.Adam(lr=0.0001)

model.compile(loss='mse', optimizer=opt)

print('training start')

bins = np.arange(-10.0,10.0,0.1)
counts = np.arange(-10.0,10.0,0.1) * 0.0

count_gt_zero = 0
count_lt_zero = 0
count_eq_zero = 0

for batch_ctr, images, sw_angles, throttle, brake_input, speeds in get_batches_multi_dir(dirs, 128):
    
    for sw_angle in sw_angles:

        if sw_angle > 0.0 or sw_angle < 0.0:
            count_lt_zero = count_lt_zero + 1
            count_gt_zero = count_gt_zero + 1
        else:
            count_eq_zero = count_eq_zero + 2

        for sw_angle in sw_angles:
            histo_loc = np.argmax(bins >= sw_angle)
            counts[histo_loc] = counts[histo_loc] + 1

        for sw_angle in sw_angles:
            histo_loc = np.argmax(bins >= -1.0 * sw_angle)
            counts[histo_loc] = counts[histo_loc] + 1

print('count_gt_zero: ', count_gt_zero)
print('count_lt_zero: ', count_lt_zero)
print('count_eq_zero: ', count_eq_zero)

fig = plt.figure()

ax=plt.subplot(111)

plt.plot(bins, counts)

ax.set_xticks(np.arange(-10,10,0.1), minor=True)
ax.set_xticks(np.arange(-10,10,1.0), minor=False)
# ax.set_yticks(np.arange(0, np.max(counts)), minor=True)

plt.grid(which='major', axis='both')
plt.grid(which='minor', axis='both')

plt.show()

for batch_ctr, images, sw_angles, throttle, brake_input, speeds in get_batches_multi_dir(dirs, 128):

    print('batch_ctr: ', batch_ctr)

    features = images[:,0,:,:]
    labels = sw_angles

    features_aug_1 = np.flip(features, 2)
    labels_aug_1 = sw_angles * -1.0
       
    # if (labels[0] > 0.01):

        # plt.subplot(211)

        # plt.title('steering angle: ' + str(labels[0]))

        # plt.imshow(features[0,:,:,:])

        # plt.subplot(212)

        # plt.title('steering angle: ' + str(labels_aug_1[0]))

        # plt.imshow(features_aug_1[0,:,:,:])

        # plt.show()

    model.fit(features, labels, validation_split=0.2, shuffle=True, epochs=4, batch_size=64)
    # model.fit(features_aug_1, labels_aug_1, validation_split=0.2, shuffle=True, epochs=4, batch_size=64)

model.save('model.h5')
