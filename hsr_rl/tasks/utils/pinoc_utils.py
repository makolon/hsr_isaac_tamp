from __future__ import print_function
import os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

def get_se3_err(pos_first, quat_first, pos_second, quat_second):
    # Retruns 6 dimensional log.SE3 error between two poses expressed as position and quaternion rotation
    
    rot_first = R.from_quat(np.array([quat_first[1],quat_first[2],quat_first[3],quat_first[0]])).as_matrix() # Quaternion in scalar last format!!!
    rot_second = R.from_quat(np.array([quat_second[1],quat_second[2],quat_second[3],quat_second[0]])).as_matrix() # Quaternion in scalar last format!!!
    
    oMfirst = pin.SE3(rot_first, pos_first)
    oMsecond = pin.SE3(rot_second, pos_second)
    firstMsecond = oMfirst.actInv(oMsecond)
    
    return pin.log(firstMsecond).vector # log gives us a spatial vector (exp co-ords)

class HSRIKSolver(object):
    def __init__(
        self,
        include_torso: bool = False, # Use torso in th IK solution
        include_base: bool = True, # Use base in th IK solution
        max_rot_vel: float = 1.0472
    ) -> None:
        # Settings
        self.damp = 1e-10
        self._include_torso = include_torso
        self._include_base = include_base
        self.max_rot_vel = max_rot_vel

        # Load urdf # TODO: fix
        urdf_file = "/root/tamp-hsr/hsr_rl/models/urdf/hsrb_description/hsrb4s.urdf"
        self.model = pin.buildModelFromUrdf(urdf_file)

        # Choose joints
        name_end_effector = "hand_palm_link"
        
        jointsOfInterest = ['arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint',
            'wrist_flex_joint', 'wrist_roll_joint']

        # Add torso joints
        if self._include_torso:
            jointsOfInterest = ['torso_lift_joint'] + jointsOfInterest

        # Add base joints
        if self._include_base:
            jointsOfInterest = ['joint_x', 'joint_y', 'joint_rz'] + jointsOfInterest

        remove_ids = list()
        for jnt in jointsOfInterest:
            if self.model.existJointName(jnt):
                remove_ids.append(self.model.getJointId(jnt))
            else:
                print('[IK WARNING]: joint ' + str(jnt) + ' does not belong to the model!')

        jointIdsToExclude = np.delete(np.arange(0, self.model.njoints), remove_ids)

        # Lock extra joints except joint 0 (root)
        reference_configuration=pin.neutral(self.model)
        if not self._include_torso:
            reference_configuration[4] = 0.1 # lock torso_lift_joint at 0.1

        self.model = pin.buildReducedModel(self.model, jointIdsToExclude[1:].tolist(),
                            reference_configuration=reference_configuration)

        assert (len(self.model.joints)==(len(jointsOfInterest)+1)), "[IK Error]: Joints != nDoFs"
        self.model_data = self.model.createData()

        # Define Joint-Limits
        self.joint_pos_min = np.array([-2.617, -1.919, -1.919, -1.919])
        self.joint_pos_max = np.array([+0.0, +3.665, +1.221, +3.665])

        if self._include_torso:
            self.joint_pos_min = np.hstack((np.array([+0.0]), self.joint_pos_min))
            self.joint_pos_max = np.hstack((np.array([+0.69]), self.joint_pos_max))
        if self._include_base:
            self.joint_pos_min = np.hstack((np.array([-10.0, -10.0, -10.0]), self.joint_pos_min))
            self.joint_pos_max = np.hstack((np.array([+10.0, +10.0, +10.0]), self.joint_pos_max))
        self.joint_pos_mid = (self.joint_pos_max + self.joint_pos_min) / 2.0

        # Get End Effector Frame ID
        self.id_EE = self.model.getFrameId(name_end_effector)

    def get_jacobian(self, curr_conf):
        # Compute Jacobian
        J = pin.computeFrameJacobian(self.model, self.model_data, curr_conf, self.id_EE)

        return J

    def solve_fk_hsr(self, curr_joints):
        pin.framesForwardKinematics(self.model, self.model_data, curr_joints)
        oMf = self.model_data.oMf[self.id_EE]
        ee_pos = oMf.translation
        ee_quat = pin.Quaternion(oMf.rotation)

        return ee_pos, np.array([ee_quat.w, ee_quat.x, ee_quat.y, ee_quat.z])
    
    def solve_ik_pos_hsr(self, des_pos, des_quat, curr_joints=None, n_trials=7, dt=0.1, pos_threshold=0.05, angle_threshold=15.*np.pi/180, verbose=True):
        # Get IK positions for hsr robot
        damp = 1e-10
        success = False

        if des_quat is not None:
            # quaternion to rot matrix
            des_rot = R.from_quat(np.array([des_quat[1], des_quat[2], des_quat[3], des_quat[0]])).as_matrix() # Quaternion in scalar last format!!!
            oMdes = pin.SE3(des_rot, des_pos)
        else:
            # 3D position error only
            des_rot = None

        if curr_joints is None:
            q = np.random.uniform(self.joint_pos_min, self.joint_pos_max)
        
        for n in range(n_trials):
            for i in range(800):
                pin.framesForwardKinematics(self.model, self.model_data, q)
                oMf = self.model_data.oMf[self.id_EE]

                if des_rot is None:
                    oMdes = pin.SE3(oMf.rotation, des_pos) # Set rotation equal to current rotation to exclude this error

                dMf = oMdes.actInv(oMf)
                err = pin.log(dMf).vector

                if (np.linalg.norm(err[0:3]) < pos_threshold) and (np.linalg.norm(err[3:6]) < angle_threshold):
                    success = True
                    break

                J = pin.computeFrameJacobian(self.model, self.model_data, q, self.id_EE)
                if des_rot is None:
                    J = J[:3, :] # Only pos errors
                    err = err[:3]

                v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q = pin.integrate(self.model, q, v*dt)

                # Clip q to within joint limits
                q = np.clip(q, self.joint_pos_min, self.joint_pos_max)

                if verbose:
                    if not i % 100:
                        print('Trial %d: iter %d: error = %s' % (n+1, i, err.T))
                    i += 1

            if success:
                best_q = np.array(q)

            else:
                # Save current solution
                best_q = np.array(q)

                # Reset q to random configuration
                q = np.random.uniform(self.joint_pos_min, self.joint_pos_max)

        if verbose:
            if success:
                print("[[[[IK: Convergence achieved!]]]")
            else:
                print("[Warning: the IK iterative algorithm has not reached convergence to the desired precision]")
        
        return success, best_q


if __name__ == '__main__':
    hsr_ik_solver = HSRIKSolver()
    success, best_q = hsr_ik_solver.solve_ik_pos_hsr(np.array([0.5, 0.1, 0.3]), np.array([0., 0., 0., 1.0]))
    print('best_q: ', best_q)