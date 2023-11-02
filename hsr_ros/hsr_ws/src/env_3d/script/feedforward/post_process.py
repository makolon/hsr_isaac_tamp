#!/usr/bin/env/python3
import numpy as np

class PlanModifier(object):
    def __init__(self):
        self.block_rigid_map = {
            'A' : 'shaft1',
            'B' : 'gear1',
            'C' : 'shaft2',
            'D' : 'gear2',
            'E' : 'gear3'
        }

    def post_process(self, action_name, args):
        """
        Modify plan using sensor data.

        Args:
            plan (list): plan is trajectory of the tamp.
            robot_pose (list): robot_pose consists of base_pose, end_effector_pose, gripper.
            rigid_poses (dict): rigid_pose consists of captured rigid body poses

        Returns:
            commands (list): commands is modified plan
        """

        if action_name == 'move_base':
            # Parse TAMP returns
            start_pose, end_pose, traj = args

            base_traj = []
            for commands in traj.commands:
                for path in commands.path:
                    base_traj.append(path.values)
            new_command = (action_name, base_traj)

        elif action_name == 'pick':
            arm, block, init_block_pose, grasp_pose, term_robot_pose, traj = args

            [traj_pick] = traj.commands
            pick_traj = []
            for path in traj_pick.path:
                pick_traj.append(path.values)

            return_traj = []
            for path in traj_pick.reverse().path:
                return_traj.append(path.values)

            new_command = (action_name, (pick_traj, return_traj))

        elif action_name == 'place':
            arm, block1, block2, init_block_pose, grasp_pose, term_robot_pose, traj = args

            [traj_place] = traj.commands
            place_traj = []
            for path in traj_place.path:
                place_traj.append(path.values)

            new_command = (action_name, (place_traj))

        elif action_name == 'insert':
            arm, block1, block2, block_pose1, block_pose2, grasp_pose, _, _, traj = args

            [traj_insert, traj_depart, traj_return] = traj.commands
            insert_traj = []
            for path in traj_insert.path:
                insert_traj.append(path.values)

            depart_traj = []
            for path in traj_depart.path:
                depart_traj.append(path.values)

            return_traj = []
            for path in traj_return.reverse().path:
                return_traj.append(path.values)

            new_command = (action_name, (insert_traj, depart_traj, return_traj))

        else:
            pass

        return new_command