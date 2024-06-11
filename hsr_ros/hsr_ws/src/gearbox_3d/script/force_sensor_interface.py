import sys
import math
import rospy
from geometry_msgs.msg import WrenchStamped


class ForceSensorInterface(object):
    def __init__(self, standalone=False):
        if standalone:
            rospy.init_node('force_torque_sensor_interface')

        self._force_data_x = 0.0
        self._force_data_y = 0.0
        self._force_data_z = 0.0

        # Subscribe force torque sensor data from HSRB
        self._wrist_wrench_sub = rospy.Subscriber(
            '/hsrb/wrist_wrench/raw', WrenchStamped, self._ft_sensor_callback
        )

        # Wait for connection
        try:
            rospy.wait_for_message('/hsrb/wrist_wrench/raw', WrenchStamped, timeout=10.0)
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)

    def _ft_sensor_callback(self, data):
        self._force_data_x = data.wrench.force.x
        self._force_data_y = data.wrench.force.y
        self._force_data_z = data.wrench.force.z

    def initialize_ft(self):
        self._force_data_x = 0.0
        self._force_data_y = 0.0
        self._force_data_z = 0.0

    def get_current_force(self):
        return [self._force_data_x, self._force_data_y, self._force_data_z]
    
    def compute_difference(self, pre_data_list, post_data_list, calc_type='l1'):
        if (len(pre_data_list) != len(post_data_list)):
            raise ValueError('Argument lists differ in length')
        
        # Calcurate square sum of difference
        if calc_type == 'l1':
            l1_sums = sum([b - a for (a, b) in zip(pre_data_list, post_data_list)])
            return l1_sums
        elif calc_type == 'l2':
            l2_sums = sum([math.pow(b - a, 2) for (a, b) in zip(pre_data_list, post_data_list)])
            return math.sqrt(l2_sums)


if __name__ == '__main__':
    ft_interface = ForceSensorInterface(standalone=True)
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        prev_ft_data = ft_interface.get_current_force()

        input('wait_for_user')

        curr_ft_data = ft_interface.get_current_force()

        force_difference = ft_interface.compute_difference(prev_ft_data, curr_ft_data)
        weight = round(force_difference / 9.81 * 1000, 1)
        print('weight:', weight)

        rate.sleep()