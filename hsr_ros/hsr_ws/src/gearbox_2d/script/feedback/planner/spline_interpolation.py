import numpy as np
from planner import Planner


class SplineInterpolationPlanner(Planner):
    def __init__(self, offset):
        super().__init__()

        # Set parameters
        self.offset = offset

    def compute_path(self, op_pose, num_steps=10):
        # Compute linear interpolation path
        assert num_steps > 1, "Too short to create waypoints"

        traj_x = np.linspace(0.0, op_pose[0], num_steps)
        traj_y = np.linspace(0.0, op_pose[1], num_steps)
        traj_z = np.linspace(0.0, op_pose[2], num_steps)
        trajectory = np.dstack((traj_x, traj_y, traj_z))

        return trajectory