import rospy
import numpy as np
from controller import Controller
from ik_solver import get_tool_pose, get_ik_generator, hsr_inverse_kinematics
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class IKController(Controller):
    def __init__(self):
        super(IKController, self).__init__()

        # Publisher
        self.arm_pub = rospy.Publisher('/hsrb/arm_trajectory_controller/command', JointTrajectory, queue_size=10)
        self.base_pub = rospy.Publisher('/hsrb/omni_base_controller/command', JointTrajectory, queue_size=10)

        # Wait for publisher has built
        while self.base_pub.get_num_connections() == 0:
            rospy.sleep(0.1)
        while self.arm_pub.get_num_connections() == 0:
            rospy.sleep(0.1)

    def set_pose(self, base_pose, joint_pose):
        base_traj = JointTrajectory()
        arm_traj = JointTrajectory()

        base_traj.joint_names = ['odom_x', 'odom_y', 'odom_t']
        arm_traj.joint_names = ['arm_lift_joint', 'arm_flex_joint',
                                'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']

        # Set base trajectory
        assert len(base_pose) == 3, "Does not match the size of base pose"
        base_p = JointTrajectoryPoint()
        base_p.positions = base_pose
        base_p.velocities = np.zeros(len(base_pose))
        base_p.time_from_start = rospy.Duration(1)
        base_traj.points = [base_p]

        # Set arm trajectory
        assert len(joint_pose) == 5, "Does not match the size of base pose"
        arm_p = JointTrajectoryPoint()
        arm_p.positions = joint_pose
        arm_p.velocities = np.zeros(len(joint_pose))
        arm_p.time_from_start = rospy.Duration(1)
        arm_traj.points = [arm_p]

        return base_traj, arm_traj

    def control(self, pose):
        # Inverse kinematics
        ik_pose = hsr_inverse_kinematics('arm', pose) # pose must be contain (pos, quat)

        if ik_pose is None:
            return
        else:
            base_pose, arm_pose = ik_pose[:3], ik_pose[3:]

        # Set target pose
        base_traj, arm_traj = self.set_pose(base_pose, arm_pose)

        # Publish target pose
        self.base_pub.publish(base_traj)
        self.arm_pub.publish(arm_traj)


if __name__ == '__main__':
    ik_controller = IKController()
    rate = rospy.Rate(50)

    pose_x = 0.0
    pose_y = 0.0
    pose_z = 0.5
    foward = (0.70710678, 0.0, 0.70710678, 0.0)
    back = (0.0, -0.70710678, 0.0, 0.70710678)
    right = (0.5, -0.5, 0.5, 0.5)
    left = (0.5, 0.5, 0.5, -0.5)
    while not rospy.is_shutdown():
        # Test forward kinematics
        fk_pose = get_tool_pose('arm')

        # Test inverse kinematics
        tool_pose = ((pose_x, pose_y, pose_z), foward)
        ik_controller.control(tool_pose)

        # Sleep
        rate.sleep()