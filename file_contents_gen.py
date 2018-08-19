# class BatchRetreiver(object):

#     def __init__(self, batch_size):
#         self.batch_size = batch_size
    
#     def __iter__(self):
#         return self

#     def 

import matplotlib.image as mpimg
import numpy as np



def get_batches_multi_dir(directories, batch_size):
    
    i = 0
    batch_ctr = 0

    sw_angles = []
    speed = []
    throttle = []
    brake_input = []
    images = []

    

    for directory in directories:

        print('opening data in ', directory)

        for l in open(directory + "/driving_log.csv", 'r'):

            data = l.split(",")

            if i == batch_size:
                
                i = 0

                batch_ctr = batch_ctr + 1

                print('----GEN_DBG: ')
                print('------batch_ctr: ', batch_ctr)

                yield batch_ctr, np.array(images), np.array(sw_angles), np.array(throttle), np.array(brake_input), np.array(speed)

                sw_angles = []
                speed = []
                throttle = []
                brake_input = []
                images = []

            sw_angles.append(float(data[3]))
            throttle.append(float(data[4]))
            brake_input.append(float(data[5]))
            speed.append(float(data[6]))
            images.append([mpimg.imread(data[0]), mpimg.imread(data[1]), mpimg.imread(data[2])])

            # print('----GEN_DBG: ')
            # print('------i: ', i)

            i = i + 1

    yield batch_ctr, np.array(images), np.array(sw_angles), np.array(throttle), np.array(brake_input), np.array(speed)

def get_batches(directory, batch_size):
    
    i = 0
    batch_ctr = 0

    sw_angles = []
    speed = []
    throttle = []
    brake_input = []
    images = []

    print('opening data in ', directory)

    for l in open(directory + "/driving_log.csv", 'r'):

        data = l.split(",")

        if i == batch_size:
            
            i = 0

            batch_ctr = batch_ctr + 1

            print('----GEN_DBG: ')
            print('------batch_ctr: ', batch_ctr)

            yield batch_ctr, np.array(images), np.array(sw_angles), np.array(throttle), np.array(brake_input), np.array(speed)

            sw_angles = []
            speed = []
            throttle = []
            brake_input = []
            images = []

        sw_angles.append(float(data[3]))
        throttle.append(float(data[4]))
        brake_input.append(float(data[5]))
        speed.append(float(data[6]))
        images.append([mpimg.imread(data[0]), mpimg.imread(data[1]), mpimg.imread(data[2])])

        # print('----GEN_DBG: ')
        # print('------i: ', i)

        i = i + 1

    yield batch_ctr, np.array(images), np.array(sw_angles), np.array(throttle), np.array(brake_input), np.array(speed)

        
def multi_dir_data_gen(dirs, batch_size, train_fraction, mode="TRAIN"):

    train_mode = mode == "TRAIN" #else assume should return validation data
    valid_mode = mode == "VALIDATION"
    all_mode = mode == "ALL"

    if not(train_mode) and not(valid_mode) and not(all_mode):
        assert mode == "VALIDATION", "mode must be either TRAIN or VALIDATION or ALL"

    for batch_ctr, images, sw_angles, throttle, brake_input, speeds in get_batches_multi_dir(dirs, 128):

        data_len = len(sw_angles)

        num_for_train = int( float( data_len ) * train_fraction )

        num_for_valid = data_len - num_for_train

        #data augmentation generates 6 times the number of images

        #perform data augmentation

        features_center = images[:,0,:,:]
        labels_center = sw_angles

        

        features_center_rev = np.flip(features_center, 2)
        labels_center_rev = sw_angles * -1.0


        features_left = images[:,1,:,:]
        labels_left = sw_angles + 0.5 # was 1.0


        features_left_rev = np.flip(features_left, 2)
        labels_left_rev = (sw_angles + 0.5)*-1.0  # was 1.0


        features_right = images[:,2,:,:]
        labels_right = sw_angles - 0.5 # was 1.0


        features_right_rev = np.flip(features_right, 2)
        labels_right_rev = (sw_angles - 0.5)*-1.0  # was 1.0

        start_index = 0
        end_index = 0

        if train_mode:
            start_index = 0
            end_index = num_for_train
        elif valid_mode:
            start_index = num_for_train
            end_index = data_len
        elif all_mode:
            start_index = 0
            end_index = data_len

        yield features_center[start_index:end_index, :, :, :], labels_center[start_index:end_index]
        yield features_center_rev[start_index:end_index, :, :, :], labels_center_rev[start_index:end_index]
        yield features_left[start_index:end_index, :, :, :], labels_left[start_index:end_index]
        yield features_left_rev[start_index:end_index, :, :, :], labels_left_rev[start_index:end_index]
        yield features_right[start_index:end_index, :, :, :], labels_right[start_index:end_index]
        yield features_right_rev[start_index:end_index, :, :, :], labels_right_rev[start_index:end_index]


# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, dirs, batch_size, split):
#         'Initialization'
#         # self.dim = dim
#         # self.batch_size = batch_size
#         # self.labels = labels
#         # self.list_IDs = list_IDs
#         # self.n_channels = n_channels
#         # self.n_classes = n_classes
#         # self.shuffle = shuffle
#         # self.on_epoch_end()

#         self.data_count = 0

#         for directory in dirs:
#             self.data_count = self.data_count + frame_count(directory)

#         self.train_gen = multi_dir_data_gen(dirs, batch_size, split, "TRAIN")
#         self.valid_gen = multi_dir_data_gen(dirs, batch_size, split, "VALIDATION")

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)

#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[i,] = np.load('data/' + ID + '.npy')

#             # Store class
#             y[i] = self.labels[ID]

#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)