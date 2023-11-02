#!/bin/env/python3
import tf
import rospy
import tf2_ros
import numpy as np
import tf2_geometry_msgs
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R


class TfManager(object):
    def __init__(self, standalone=False):
        if standalone:
            rospy.init_node('tf_manager')

        # TF listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # Publisher
        self.tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=1)
    
    def get_link_pose(self, link_name='hand_palm_link'):
        # Translate from map coordinate to arbitrary coordinate of robot.
        ee_pose = tf2_geometry_msgs.PoseStamped()
        ee_pose.header.frame_id = link_name
        ee_pose.header.stamp = rospy.Time(0)
        ee_pose.pose.orientation.w = 1.0

        try:
            # Get transform at current time
            global_pose = self.tfBuffer.transform(ee_pose, 'map')

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            return None

        return global_pose

    def get_init_pose(self, mocap_poses):
        # Get red_shaft pose
        gearbox_x = mocap_poses['green_gear'].pose.position.x
        gearbox_y = mocap_poses['green_gear'].pose.position.y
        gearbox_z = mocap_poses['green_gear'].pose.position.z

        # Get current end effector pose
        current_ee_pose_x = mocap_poses['end_effector'].pose.position.x
        current_ee_pose_y = mocap_poses['end_effector'].pose.position.y
        current_ee_pose_z = mocap_poses['end_effector'].pose.position.z

        # Culculate difference pose
        diff_x_pose = current_ee_pose_x - gearbox_x
        diff_y_pose = current_ee_pose_y - gearbox_y
        diff_z_pose = current_ee_pose_z - gearbox_z

        init_base_pose = np.array([diff_x_pose, diff_y_pose, diff_z_pose])

        return init_base_pose

    # Transform target_pose on end effector coordinate to map coordinate
    def transform_coordinate(self, target_pose):
        # Initialize vector and matrix
        trans_mat = np.zeros((4, 4))
        qc_trans = np.zeros(4)

        # Initialize map coordination
        qc_trans[0] = target_pose[0]
        qc_trans[1] = target_pose[1]
        qc_trans[2] = target_pose[2]
        qc_trans[3] = 1.0

        ee_pose = self.get_link_pose()
        if ee_pose is None:
            return

        # Initialize ee coordination
        translation = np.array([
            ee_pose.pose.position.x,
            ee_pose.pose.position.y,
            ee_pose.pose.position.z
        ])
        rot = np.array([
            ee_pose.pose.orientation.x,
            ee_pose.pose.orientation.y,
            ee_pose.pose.orientation.z,
            ee_pose.pose.orientation.w
        ])
        rot_mat = R.from_quat(rot).as_matrix()

        # Calculate translation
        trans_mat[3:, 3:] = 1.0
        trans_mat[:3, 3] = translation
        trans_mat[:3, :3] = rot_mat

        qw_trans = trans_mat @ qc_trans
        qw_trans = qw_trans[:3]

        # Calculate rotation
        qw_rot = np.array([
            ee_pose.pose.orientation.x,
            ee_pose.pose.orientation.y,
            ee_pose.pose.orientation.z,
            ee_pose.pose.orientation.w
        ])

        return np.concatenate([qw_trans, qw_rot])

    def publish_mocap_to_map(self, mocap_poses):
        tf_list = []
        for rigid_name, mocap_pose in mocap_poses.items():
            t = TransformStamped()
            t.header.frame_id = 'map'
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = rigid_name

            t.transform.translation.x = mocap_pose.pose.position.x
            t.transform.translation.y = mocap_pose.pose.position.y
            t.transform.translation.z = mocap_pose.pose.position.z
            t.transform.rotation.x = mocap_pose.pose.rotation.x
            t.transform.rotation.y = mocap_pose.pose.rotation.y
            t.transform.rotation.z = mocap_pose.pose.rotation.z
            t.transform.rotation.w = mocap_pose.pose.rotation.w

            tf_list.append(t)

        tfm = TFMessage(tf_list)

        # Publish tf message
        self.tf_pub.publish(tfm)


if __name__ == '__main__':
    tf_manager = TfManager(standalone=True)
    while not rospy.is_shutdown():
        transform = tf_manager.get_link_pose('hand_palm_link')

        test_pose1 = np.array([-0.5, 0.0, 0.0]) # -0.5 to z direction on map
        test_pose1 = tf_manager.transform_coordinate(test_pose1)

        test_pose2 = np.array([0.0, 1.0, 0.0]) # -1.0 to x direction on map
        test_pose2 = tf_manager.transform_coordinate(test_pose2)

        test_pose3 = np.array([-0.3, -1.0, 0.0])
        test_pose3 = tf_manager.transform_coordinate(test_pose3)

        ee_pose = tf_manager.get_link_pose()
        if ee_pose is None:
            continue
        test_pose4 = np.array([
            ee_pose.pose.position.x,
            ee_pose.pose.position.y,
            ee_pose.pose.position.z
        ])
        test_pose4 = tf_manager.transform_coordinate(test_pose4)

        test_pose5 = np.zeros(3)
        test_pose5 = tf_manager.transform_coordinate(test_pose5)

        print('test_1: ', test_pose1)
        print('test_2: ', test_pose2)
        print('test_3: ', test_pose3)
        print('test_4: ', test_pose4)
        print('test_5: ', test_pose5)