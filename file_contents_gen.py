# class BatchRetreiver(object):

#     def __init__(self, batch_size):
#         self.batch_size = batch_size
    
#     def __iter__(self):
#         return self

#     def 

import matplotlib.image as mpimg
import numpy as np


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
        images.append([mpimg.imread(data[0])])#, mpimg.imread(data[1]), mpimg.imread(data[2])])

        print('----GEN_DBG: ')
        print('------i: ', i)

        i = i + 1
    
    yield batch_ctr, np.array(images), np.array(sw_angles), np.array(throttle), np.array(brake_input), np.array(speed)

        
