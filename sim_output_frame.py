import matplotlib.image as mpimg

class SimOutputFrame:

    def __init__(self, csv_line):
        l = csv_line.split(",")
        assert len(l) == 7, "expected each data member to have length 7"
        self.image_center = mpimg.imread(l[0])
        self.image_left = mpimg.imread(l[1])
        self.image_right = mpimg.imread(l[2])
        self.steering_angle = float(l[3])
        self.entry_4 = float(l[4])
        self.entry_5 = float(l[5])
        self.speed = float(l[6])
    


