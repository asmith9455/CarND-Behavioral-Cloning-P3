from data_exploration import explore_over_time, frame_count, generate_summary_plot

from file_contents_gen import get_batches_multi_dir, multi_dir_data_gen

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

# choose the operations to perform

# load_prev_model can be combined with train_model to 'add on' to the knowledge of the network
produce_graph = True
load_prev_model = True 
train_model = True # train the model using the data in the dirs variable
summary_plot = False # generate a matplotlib figure that include plots of steering angle, throttle, braking, etc and sample images from the 3 cameras
compile_statistics = False # generate statistics that indicate the distribution of the data by steering angle

dirs = \
[
    "../data/P3-sim-data-udacity/data",
    "../data/P3-sim-data-hard-left-0"
]

for d in dirs:
    print('frame count for', d, 'is: ', frame_count(d))

if summary_plot:

    images, sw_angles, throttle, brake_input, speeds = explore_over_time(fname, 300)

    generate_summary_plot(images, sw_angles, throttle, brake_input, speeds)

if train_model:

    model = Sequential()    # use the keras Sequential model type

    image_shape = (70, 160, 3)# images[0,0,:,:].shape

    # model.add(__import__('tensorflow').keras.layers.InputLayer(input_shape=(None, 160, 320, 3)))

    # started with the NVIDIA End-to-End SDC network described here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

    # made adjustments to the sizes of the layers by trial and error and used greyscale instead of colour images

    model.add(Lambda(lambda x: __import__('tensorflow').image.rgb_to_grayscale(x)))

    # crop out parts of the top and bottom of the image, since these parts of the image do not seem necessary
    # for steering the car.
    model.add(Cropping2D(cropping=( (60,25), (0,0) )))

    # use a keras Lambda to resize the image
    model.add(Lambda(lambda x: __import__('keras').backend.tf.image.resize_images(x, (50,160))))

    # change the range of the data to [-1.0, 1.0]
    model.add(Lambda(lambda x: (x / 255.0 - 0.5) * 2))

    # add the convolutional layers 
    model.add(Conv2D(filters=12, kernel_size=5, strides=(1,1), activation='relu'))

    model.add(Conv2D(filters=24, kernel_size=5, strides=(2,2), activation='relu'))

    model.add(Conv2D(filters=36, kernel_size=5, strides=(2,2), activation='relu'))

    model.add(Conv2D(filters=48, kernel_size=3, strides=(1,1), activation='relu'))

    model.add(Conv2D(filters=64, kernel_size=3, strides=(1,1), activation='relu'))

    # flatten the convolutional layers to connect to the Fully Connected layers
    model.add(Flatten())

    model.add(Dense(400, activation='relu'))

    model.add(Dense(600, activation='relu'))

    model.add(Dense(300, activation='relu'))

    model.add(Dense(100, activation='relu'))

    # use dropout to improve generalization to other data

    model.add(Dropout(0.5))

    model.add(Dense(1)) #steering wheel angle is the output
    # features = images[:,0,:,:]
    # labels = sw_angles

    opt = keras.optimizers.Adam(lr=0.0001)  # use the Adam Optimizer - was successful in P2 and worked well here too

    # get the 'generator' for the data
    # In the multi_dir_data_gen function, I included an option to split the data into Training and Validation data
    # the keras fit function also provides options to split data into training/validation sets
    data_gen_all = multi_dir_data_gen(dirs, 64, 0.2, "ALL")
    # data_gen_train = multi_dir_data_gen(dirs, 64, 0.2, "TRAIN")
    # data_gen_valid = multi_dir_data_gen(dirs, 64, 0.2, "VALIDATION")

    model.compile(loss='mse', optimizer=opt) 

    if load_prev_model:
        model = keras.models.load_model('model.h5')

    if produce_graph:
        print(model.summary())
        from keras.utils import plot_model
        plot_model(model, to_file='model.png', show_shapes=True)
        exit()

    # I attempted to use model.fit_generator but there were some problems 
    # using my data generator with custom batch size and the normal fit function from keras
    # works well anyway

    for features, labels in data_gen_all:
        print('features shape: ', features.shape)
        print('labels shape: ', labels.shape)
        model.fit(features, labels, validation_split=0.2, shuffle=True, epochs=5, batch_size=64)
    
    # save the model for later recall
    model.save('model.h5')

if compile_statistics:

    #define an array of bin boundaries and an array of counts (initialized to 0)
    bins = np.arange(-10.0,10.0,0.1)
    counts = np.arange(-10.0,10.0,0.1) * 0.0

    # count greater than, less than and equal to 0 steering angles to validate the data augmentation that is built into the generator
    count_gt_zero = 0
    count_lt_zero = 0
    count_eq_zero = 0

    # this loop generates the histogram counts
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

    # plot the histogram
    fig = plt.figure()

    ax=plt.subplot(111)

    plt.plot(bins, counts)

    ax.set_xticks(np.arange(-10,10,0.1), minor=True)
    ax.set_xticks(np.arange(-10,10,1.0), minor=False)
    # ax.set_yticks(np.arange(0, np.max(counts)), minor=True)

    plt.grid(which='major', axis='both')
    plt.grid(which='minor', axis='both')

    plt.show()

# model.fit_generator(data_gen_train, validation_data=data_gen_valid, samples_per_epoch=10, epochs=10)

# //steering: -1 to 1
# // throttle 0 to 1
# // brake 0 1
# // speed 0 30
               


