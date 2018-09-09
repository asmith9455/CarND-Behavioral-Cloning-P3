import matplotlib.image as mpimg
import numpy as np



def get_batches_multi_dir(directories, batch_size):
    # get batches from multiple (a list of) directories

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
    # get batches from a single directory

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

    # same idea as get_batches_multi_dir, but augments the training data 
    # and has the option to perform training/validation split, but without 
    # total shuffling for disk access speed reasons

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
