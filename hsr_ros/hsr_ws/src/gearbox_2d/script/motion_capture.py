#!/bin/env/python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

class MocapInterface(object):
    def __init__(self, standalone=False):
        if standalone:
            rospy.init_node('mocap_interface')

        # Mocap list
        self.rigid_body_list = (
            "base",
            "blue_gear",
            "green_gear",
            "red_gear",
            "yellow_shaft",
            "red_shaft",
            "end_effector"
        )

        # Mocap to map
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        # Rigid body pose dictionary
        self.rigid_posestamped = {
            name: PoseStamped() for name in self.rigid_body_list
        }

        # Subscriber
        self._mocap_sub = [
            rospy.Subscriber('/mocap_pose_topic/{0}_pose'.format(self.rigid_body_list[i]), PoseStamped, self._mocap_cb, callback_args=i)
                for i in range(len(self.rigid_body_list))
        ]

    def _mocap_cb(self, mocap_data, id):
        converted_pose = self.convert_mocap_to_map(mocap_data)
        mocap_data.header.frame_id = 'map'
        mocap_data.pose.position.x = converted_pose[0]
        mocap_data.pose.position.y = converted_pose[1]
        mocap_data.pose.position.z = converted_pose[2]
        mocap_data.pose.orientation.x = converted_pose[3]
        mocap_data.pose.orientation.y = converted_pose[4]
        mocap_data.pose.orientation.z = converted_pose[5]
        mocap_data.pose.orientation.w = converted_pose[6]

        self.rigid_posestamped[self.rigid_body_list[id]] = mocap_data

    def get_pose(self, name):
        return self.rigid_posestamped[name]

    def get_poses(self):
        return self.rigid_posestamped
    
    def convert_mocap_to_map(self, mocap_data):
        # Initialize vector and matrix
        trans_mat = np.zeros((4, 4))
        qc_trans = np.zeros(4)
        qc_rot = np.zeros(4)

        # Calculate translation
        qc_trans[0] = mocap_data.pose.position.x
        qc_trans[1] = mocap_data.pose.position.y
        qc_trans[2] = mocap_data.pose.position.z
        qc_trans[3] = 1.0

        trans_mat[3:, 3:] = 1.0
        trans_mat[:3, 3] = self.translation
        trans_mat[:3, :3] = self.rotation

        qw_trans = trans_mat @ qc_trans
        qw_trans = qw_trans[:3]

        # Calculate rotation
        qc_rot[0] = mocap_data.pose.orientation.x
        qc_rot[1] = mocap_data.pose.orientation.y
        qc_rot[2] = mocap_data.pose.orientation.z
        qc_rot[3] = mocap_data.pose.orientation.w

        qc_rot_mat = R.from_quat(qc_rot).as_matrix()

        qw_rot_mat = self.rotation @ qc_rot_mat
        qw_rot = R.from_matrix(qw_rot_mat).as_quat()

        return np.concatenate([qw_trans, qw_rot])


if __name__ == '__main__':
    mocap_interface = MocapInterface(standalone=True)
    while not rospy.is_shutdown():
        # Test each rigid body
        base_pose = mocap_interface.get_pose('base')
        print('base_pose: ', base_pose)
        blue_gear_pose = mocap_interface.get_pose('blue_gear')
        print('blue_gear_pose: ', blue_gear_pose)
        green_gear_pose = mocap_interface.get_pose('green_gear')
        print('green_gear_pose: ', green_gear_pose)
        red_gear_pose = mocap_interface.get_pose('red_gear')
        print('red_gear_pose: ', red_gear_pose)
        yellow_shaft_pose = mocap_interface.get_pose('yellow_shaft')
        print('yellow_shaft_pose: ', yellow_shaft_pose)
        red_shaft_pose = mocap_interface.get_pose('red_shaft')
        print('red_shaft_pose: ', red_shaft_pose)
        end_effector_pose = mocap_interface.get_pose('end_effector')
        print('end_effector_pose: ', end_effector_pose)

        # Test all rigid bodies
        rigid_poses = mocap_interface.get_poses()
        print('rigid_poses: ', rigid_poses['blue_gear'])