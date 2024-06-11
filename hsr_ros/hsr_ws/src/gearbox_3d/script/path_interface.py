#!/usr/bin/env/python3
import numpy as np

class PlanModifier(object):

    def post_process(self, action_name, object_names, args):
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

            [traj_move] = traj.commands
            base_traj = []
            for path in traj_move.path:
                base_traj.append(path.values)

            object_name = None

            new_command = (action_name, object_name, base_traj)

        elif action_name == 'pick':
            arm, block, init_block_pose, grasp_pose, term_robot_pose, traj = args

            [traj_approach, traj_pick, traj_return] = traj.commands
            pick_traj = []
            for path in traj_approach.path:
                pick_traj.append(path.values)

            for path in traj_pick.path:
                pick_traj.append(path.values)

            return_traj = []
            for path in traj_pick.reverse().path:
                return_traj.append(path.values)
            for path in traj_return.path:
                return_traj.append(path.values)

            object_name = object_names[block]

            new_command = (action_name, object_name, (pick_traj, return_traj))

        elif action_name == 'place':
            arm, block1, block2, init_block_pose, grasp_pose, term_robot_pose, traj = args

            [traj_approach, traj_place] = traj.commands
            place_traj = []
            for path in traj_approach.path:
                place_traj.append(path.values)

            for path in traj_place.path:
                place_traj.append(path.values)

            object_name = object_names[block1]

            new_command = (action_name, object_name, (place_traj))

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

            object_name = object_names[block1] # TODO: check whether block1 is correct

            new_command = (action_name, object_name, (insert_traj, depart_traj, return_traj))

        else:
            pass

        return new_command