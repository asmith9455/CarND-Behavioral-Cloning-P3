from sim_output_frame import SimOutputFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def generate_summary_plot(images, sw_angle, throttle, brake_input, speeds):

    plt.figure()

    plt.subplot(321)

    plt.title('sw angle over time')

    plt.plot(sw_angle)

    plt.subplot(323)

    plt.title('speed over time image')

    plt.plot(speeds)

    plt.subplot(325)

    plt.title('entry 4 and entry 5')

    plt.plot(throttle, label="throttle") 

    plt.plot(brake_input, label="brake_input")

    plt.legend(loc="upper right")

    plt.subplot(322)

    plt.title('center image')

    plt.imshow(images[0,0,:,:])

    plt.subplot(324)

    plt.title('left image')

    plt.imshow(images[0,1,:,:])

    plt.subplot(326)

    plt.title('right image')

    plt.imshow(images[0,2,:,:])

    plt.show()

def frame_count(directory):

    c = 0

    for l in open(directory + "/driving_log.csv", 'r'):
        c = c + 1
    
    return c

def explore_over_time(directory, N):

    sw_angles = []
    speed = []
    throttle = []
    brake_input = []
    images = []

    print('starting data read...')

    i = int(0)

    for l in open(directory + "/driving_log.csv", 'r'):

        if i >= N:
            break

        data = l.split(",")

        sw_angles.append(float(data[3]))
        throttle.append(float(data[4]))
        brake_input.append(float(data[5]))
        speed.append(float(data[6]))
        images.append([mpimg.imread(data[0]), mpimg.imread(data[1]), mpimg.imread(data[2])])

        i = i + 1

    print('finished data read')

    # plt.plot(sw_angles)

    # plt.show()

    return np.array(images), np.array(sw_angles), np.array(throttle), np.array(brake_input), np.array(speed)

def explore_sw_angles_over_time(directory, N):

    sw_angles = []

    print('starting data read...')

    i = int(0)

    for l in open(directory + "/driving_log.csv", 'r'):

        if i >= N:
            break

        data = l.split(",")

        sw_angles.append(float(data[3]))

        i = i + 1

    print('finished data read')

    # plt.plot(sw_angles)

    # plt.show()

    return np.array(sw_angles)

def explore_speed_over_time(directory, N):

    speed = []

    print('starting data read...')

    i = int(0)

    for l in open(directory + "/driving_log.csv", 'r'):

        if i >= N:
            break

        data = l.split(",")

        speed.append(float(data[6]))

        i = i + 1

    print('finished data read')

    # plt.plot(speed)

    # plt.show()

    return np.array(speed)

def explore_images_over_time(directory, N):

    images = []

    print('starting data read...')

    i = int(0)

    for l in open(directory + "/driving_log.csv", 'r'):

        if i >= N:
            break

        data = l.split(",")

        images.append([mpimg.imread(data[0]), mpimg.imread(data[1]), mpimg.imread(data[2])])

        i = i + 1

    print('finished data read')

    return np.array(images)


def explore_data(directory, N):

    sim_data = []

    print('starting data read...')

    i = int(0)

    for l in open(directory + "/driving_log.csv", 'r'):

        if i >= N:
            break

        sim_data.append(SimOutputFrame(l))

        i = i + 1

    print('finished data read')
    
