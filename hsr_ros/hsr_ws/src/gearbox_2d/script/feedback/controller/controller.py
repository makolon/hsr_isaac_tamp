import rospy
import numpy as np


class Controller(object):
    def __init__(self):
        pass

    def set_pose(self):
        raise NotImplementedError("Implement set_pose method")

    def control(self):
        raise NotImplementedError("Implement control method")
