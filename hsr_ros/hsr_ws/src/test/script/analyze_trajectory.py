import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class AnalyzeTrajectory(object):
    def __init__(self, mode='both', joint_index=None):
        # Load ee trajectory
        self.sim_ee_traj = np.load(f"log/simulation_ee_traj.npy", allow_pickle=True)
        self.measured_ee_traj = np.load(f"log/measured_ee_traj.npy", allow_pickle=True)

        # Load joint trajectory
        self.sim_joint_traj = np.load(f"log/simulation_joint_traj.npy", allow_pickle=True)
        self.measured_joint_traj = np.load(f"log/measured_joint_traj.npy", allow_pickle=True)

        # Length of ee/joint trajectory
        self.ee_traj_len = min(len(self.sim_ee_traj), len(self.measured_ee_traj))
        self.jt_traj_len = min(len(self.sim_joint_traj), len(self.measured_joint_traj))

        # Visualization settings
        self.fps = 8
        self.num_joints = 8
        self.mode = mode
        self.joint_index = joint_index
        self.joint_name = {'2': 'BaseRotation', '3': 'ArmLift', '4': 'ArmFlex', 
                           '5': 'ArmRoll', '6': 'WristFlex', '7': 'WristRoll'}
        if self.joint_index is None:
            self.vis_all = True
        else:
            self.vis_all = False

    def visualize_ee_3d_traj(self):
        # Visualize end effector trajectory
        animation_duration = self.ee_traj_len
        self.time_steps_ee_3d = np.linspace(0, animation_duration, animation_duration * self.fps)

        fig = plt.figure()
        self.ax_ee_3d = fig.add_subplot(111, projection='3d')
        self.ax_ee_3d.view_init(elev=45, azim=75)
        ani = FuncAnimation(fig, self.update_ee_3d_traj_animation, frames=animation_duration, interval=1000/self.fps)

        # Save fig
        ani.save(f"{self.mode}_end_effector_3d_trajectory.gif", writer='imagemagick', fps=10)
        ani.save(f"{self.mode}_end_effector_3d_trajectory.mp4", writer='ffmpeg', fps=10)

    def visualize_ee_2d_traj(self, dir='x'):
        # Visualize end effector trajectory
        animation_duration = self.ee_traj_len 
        self.time_steps_ee_2d = np.linspace(0, animation_duration, animation_duration * self.fps)

        fig = plt.figure()
        self.ax_ee_2d = fig.add_subplot(1, 1, 1)
        ani = FuncAnimation(fig, self.update_ee_2d_traj_animation, frames=animation_duration, interval=1000/self.fps)

        # Save fig
        ani.save(f"{self.mode}_end_effector_2d_trajectory.gif", writer='imagemagick', fps=10)
        ani.save(f"{self.mode}_end_effector_2d_trajectory.mp4", writer='ffmpeg', fps=10)

    def visualize_joint_2d_traj(self):
        # Visualize each joint trajectory
        animation_duration = self.jt_traj_len
        self.time_steps_jt_2d = np.linspace(0, animation_duration, animation_duration * self.fps)

        if self.vis_all:
            fig, self.ax_jt = plt.subplots(self.num_joints, 1, sharex=True)
            ani = FuncAnimation(fig, self.update_all_joint_2d_traj_animation, frames=animation_duration, interval=1000/self.fps)
        else:
            fig = plt.figure()
            self.ax_jt = fig.add_subplot(1, 1, 1)
            ani = FuncAnimation(fig, self.update_joint_2d_traj_animation, frames=animation_duration, interval=1000/self.fps)

        # Save fig
        if self.vis_all:
            ani.save(f"{self.mode}_all_joint_2d_trajectory.gif", writer='imagemagick', fps=10)
            ani.save(f"{self.mode}_all_joint_2d_trajectory.mp4", writer='ffmpeg', fps=10)
        else:
            ani.save(f"{self.mode}_joint_{self.joint_index}_2d_trajectory.gif", writer='imagemagick', fps=10)
            ani.save(f"{self.mode}_joint_{self.joint_index}_2d_trajectory.mp4", writer='ffmpeg', fps=10)

    def update_ee_3d_traj_animation(self, i):
        self.ax_ee_3d.cla()
        self.ax_ee_3d.set_xlim(0.5, 1.5)
        self.ax_ee_3d.set_ylim(-1.0, 1.0)
        self.ax_ee_3d.set_zlim(0.0, 1.0)
        self.ax_ee_3d.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        self.ax_ee_3d.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        self.ax_ee_3d.set_xlabel("X")
        self.ax_ee_3d.set_ylabel("Y")
        self.ax_ee_3d.set_zlabel("Z")
        self.ax_ee_3d.set_title(f"Time: {self.time_steps_ee_3d[i]:.2f}s")

        if self.mode == 'simulation':
            ### Simulation
            position = self.sim_ee_traj[i][0]
            orientation = self.sim_ee_traj[i][1]

            rotation_matrix = R.from_quat(orientation).as_matrix()
            self.ax_ee_3d.scatter(*position, color='b', marker='*', s=3)

            axis_x_length, axis_y_length, axis_z_length = 0.075, 0.1, 0.05
            x_axis = position + axis_x_length * rotation_matrix[:, 0]
            y_axis = position + axis_y_length * rotation_matrix[:, 1]
            z_axis = position + axis_z_length * rotation_matrix[:, 2]
            self.ax_ee_3d.plot([position[0], x_axis[0]], [position[1], x_axis[1]], [position[2], x_axis[2]], color='r', linewidth='3')
            self.ax_ee_3d.plot([position[0], y_axis[0]], [position[1], y_axis[1]], [position[2], y_axis[2]], color='g', linewidth='3')
            self.ax_ee_3d.plot([position[0], z_axis[0]], [position[1], z_axis[1]], [position[2], z_axis[2]], color='b', linewidth='3')

            traj = self.sim_ee_traj[:i+1, 0]

            x = [point[0] for point in traj]
            y = [point[1] for point in traj]
            z = [point[2] for point in traj]
            self.ax_ee_3d.plot(x, y, z, color='deepskyblue', linewidth=1, marker='*', markersize=2)

        elif self.mode == 'measured':
            ### Measured
            position = self.measured_ee_traj[i][0]
            orientation = self.measured_ee_traj[i][1]

            rotation_matrix = R.from_quat(orientation).as_matrix()
            self.ax_ee_3d.scatter(*position, color='g', marker='^', s=3)

            axis_x_length, axis_y_length, axis_z_length = 0.075, 0.1, 0.05
            x_axis = position + axis_x_length * rotation_matrix[:, 0]
            y_axis = position + axis_y_length * rotation_matrix[:, 1]
            z_axis = position + axis_z_length * rotation_matrix[:, 2]
            self.ax_ee_3d.plot([position[0], x_axis[0]], [position[1], x_axis[1]], [position[2], x_axis[2]], color='r', linewidth='3')
            self.ax_ee_3d.plot([position[0], y_axis[0]], [position[1], y_axis[1]], [position[2], y_axis[2]], color='g', linewidth='3')
            self.ax_ee_3d.plot([position[0], z_axis[0]], [position[1], z_axis[1]], [position[2], z_axis[2]], color='b', linewidth='3')

            traj = self.measured_ee_traj[:i+1, 0]

            x = [point[0] for point in traj]
            y = [point[1] for point in traj]
            z = [point[2] for point in traj]
            self.ax_ee_3d.plot(x, y, z, color='limegreen', linewidth=1, marker='^', markersize=2)

        elif self.mode == 'both':
            ### Simulation
            position = self.sim_ee_traj[i][0]
            orientation = self.sim_ee_traj[i][1]

            rotation_matrix = R.from_quat(orientation).as_matrix()
            self.ax_ee_3d.scatter(*position, color='b', marker='*', s=3)

            axis_x_length, axis_y_length, axis_z_length = 0.075, 0.1, 0.05
            x_axis = position + axis_x_length * rotation_matrix[:, 0]
            y_axis = position + axis_y_length * rotation_matrix[:, 1]
            z_axis = position + axis_z_length * rotation_matrix[:, 2]
            self.ax_ee_3d.plot([position[0], x_axis[0]], [position[1], x_axis[1]], [position[2], x_axis[2]], color='r', linewidth='3')
            self.ax_ee_3d.plot([position[0], y_axis[0]], [position[1], y_axis[1]], [position[2], y_axis[2]], color='g', linewidth='3')
            self.ax_ee_3d.plot([position[0], z_axis[0]], [position[1], z_axis[1]], [position[2], z_axis[2]], color='b', linewidth='3')

            traj = self.sim_ee_traj[:i+1, 0]

            x = [point[0] for point in traj]
            y = [point[1] for point in traj]
            z = [point[2] for point in traj]
            self.ax_ee_3d.plot(x, y, z, color='deepskyblue', linewidth=1, marker='*', markersize=2)

            ### Measured
            position = self.measured_ee_traj[i][0]
            orientation = self.measured_ee_traj[i][1]

            rotation_matrix = R.from_quat(orientation).as_matrix()
            self.ax_ee_3d.scatter(*position, color='g', marker='^', s=3)

            axis_x_length, axis_y_length, axis_z_length = 0.075, 0.1, 0.05
            x_axis = position + axis_x_length * rotation_matrix[:, 0]
            y_axis = position + axis_y_length * rotation_matrix[:, 1]
            z_axis = position + axis_z_length * rotation_matrix[:, 2]
            self.ax_ee_3d.plot([position[0], x_axis[0]], [position[1], x_axis[1]], [position[2], x_axis[2]], color='r', linewidth='3')
            self.ax_ee_3d.plot([position[0], y_axis[0]], [position[1], y_axis[1]], [position[2], y_axis[2]], color='g', linewidth='3')
            self.ax_ee_3d.plot([position[0], z_axis[0]], [position[1], z_axis[1]], [position[2], z_axis[2]], color='b', linewidth='3')

            traj = self.measured_ee_traj[:i+1, 0]

            x = [point[0] for point in traj]
            y = [point[1] for point in traj]
            z = [point[2] for point in traj]
            self.ax_ee_3d.plot(x, y, z, color='limegreen', linewidth=1, marker='^', markersize=2)

    def update_ee_2d_traj_animation(self, i):
        self.ax_ee_2d.cla()
        self.ax_ee_2d.set_xlim(0, self.ee_traj_len)
        self.ax_ee_2d.set_ylim(0.0, 1.5)
        self.ax_ee_2d.set_xlabel("Time (s)")
        self.ax_ee_2d.set_ylabel("End Effector Value (rad)")
        self.ax_ee_2d.set_title(f"Joint Values over Time")

        self.sim_ee_traj.T
        self.measured_ee_traj.T

        ### Simulation / Measured
        for idx in self.joint_idxs:
            for joint_values in self.sim_ee_traj[idx*len(self.joint_idxs)]:
                self.ax_ee_2d.plot(self.time_steps_jt_2d[:i+1], joint_values[:i+1])

            for joint_values in self.measured_joint_traj[idx*len(self.joint_idxs)]:
                self.ax_ee_2d.plot(self.time_steps_jt_2d[:i+1], joint_values[:i+1])

        self.ax_ee_2d.legend(loc="upper left")

    def update_all_joint_2d_traj_animation(self, i):
        plot_length = self.jt_traj_len / self.fps

        for joint_index in range(self.num_joints):
            if joint_index == 0:
                min_range, max_range = 0.0, 1.5
                self.ax_jt[joint_index].cla()
                self.ax_jt[joint_index].grid()
                self.ax_jt[joint_index].set_xlim(0, plot_length)
                self.ax_jt[joint_index].set_ylim(min_range, max_range)
                self.ax_jt[joint_index].set_ylabel("X Position (m)", fontsize=8)
                self.ax_jt[joint_index].set_title("Joint Values over Time", fontsize=14)
            elif joint_index == 1:
                min_range, max_range = -1.0, 1.0
                self.ax_jt[joint_index].cla()
                self.ax_jt[joint_index].grid()
                self.ax_jt[joint_index].set_xlim(0, plot_length)
                self.ax_jt[joint_index].set_ylabel("Y Position (m)", fontsize=8)
                self.ax_jt[joint_index].set_ylim(min_range, max_range)
            elif joint_index == 2:
                min_range, max_range = -np.pi, np.pi
                y_label = f"{self.joint_name[str(joint_index)]} Value(rad)"
                self.ax_jt[joint_index].cla()
                self.ax_jt[joint_index].grid()
                self.ax_jt[joint_index].set_xlim(0, plot_length)
                self.ax_jt[joint_index].set_ylim(min_range, max_range)
                self.ax_jt[joint_index].set_ylabel(y_label, fontsize=8)
            elif joint_index == 3:
                min_range, max_range = -0.1, 0.5
                y_label = f"{self.joint_name[str(joint_index)]} Value(m)"
                self.ax_jt[joint_index].cla()
                self.ax_jt[joint_index].grid()
                self.ax_jt[joint_index].set_xlim(0, plot_length)
                self.ax_jt[joint_index].set_ylim(min_range, max_range)
                self.ax_jt[joint_index].set_ylabel(y_label, fontsize=8)
            elif joint_index in (4, 5, 6):
                min_range, max_range = -np.pi, np.pi
                y_label = f"{self.joint_name[str(joint_index)]} Value(rad)"
                self.ax_jt[joint_index].cla()
                self.ax_jt[joint_index].grid()
                self.ax_jt[joint_index].set_xlim(0, plot_length)
                self.ax_jt[joint_index].set_ylim(min_range, max_range)
                self.ax_jt[joint_index].set_ylabel(y_label, fontsize=8)
            elif joint_index == 7:
                min_range, max_range = -np.pi, np.pi
                y_label = f"{self.joint_name[str(joint_index)]} Value(rad)"
                self.ax_jt[joint_index].cla()
                self.ax_jt[joint_index].grid()
                self.ax_jt[joint_index].set_xlim(0, plot_length)
                self.ax_jt[joint_index].set_ylim(min_range, max_range)
                self.ax_jt[joint_index].set_ylabel(y_label, fontsize=8)
                self.ax_jt[joint_index].set_xlabel("Time (s)", fontsize=12)

            ### Simulation
            traj = self.sim_joint_traj[:i, joint_index]
            self.ax_jt[joint_index].plot(self.time_steps_jt_2d[:i], traj, linestyle='dashed', label='simulation')
            
            ### Measured
            traj = self.measured_joint_traj[:i, joint_index]
            self.ax_jt[joint_index].plot(self.time_steps_jt_2d[:i], traj, linestyle='dashdot', label='measured')

            self.ax_jt[joint_index].legend(loc="upper right", fontsize=12)

    def update_joint_2d_traj_animation(self, i):
        plot_length = self.jt_traj_len / self.fps

        if self.joint_index == 0:
            min_range, max_range = 0.0, 1.5
            self.ax_jt.cla()
            self.ax_jt.grid()
            self.ax_jt.set_xlim(0, plot_length)
            self.ax_jt.set_ylim(min_range, max_range)
            self.ax_jt.set_xlabel("Time (s)", fontsize=12)
            self.ax_jt.set_ylabel("X Position (m)", fontsize=12)
            self.ax_jt.set_title("Joint Values over Time", fontsize=14)
        elif self.joint_index == 1:
            min_range, max_range = -1.0, 1.0
            self.ax_jt.cla()
            self.ax_jt.grid()
            self.ax_jt.set_xlim(0, plot_length)
            self.ax_jt.set_ylim(min_range, max_range)
            self.ax_jt.set_xlabel("Time (s)", fontsize=12)
            self.ax_jt.set_ylabel("Y Position Value (m)", fontsize=12)
            self.ax_jt.set_title("Joint Values over Time", fontsize=14)
        elif self.joint_index == 3:
            min_range, max_range = -0.1, 0.5
            self.ax_jt.cla()
            self.ax_jt.grid()
            self.ax_jt.set_xlim(0, plot_length)
            self.ax_jt.set_ylim(min_range, max_range)
            self.ax_jt.set_xlabel("Time (s)", fontsize=12)
            self.ax_jt.set_ylabel("ArmLift Value (m)", fontsize=12)
            self.ax_jt.set_title("Joint Values over Time", fontsize=14)
        else:
            min_range, max_range = -np.pi, np.pi
            self.ax_jt.cla()
            self.ax_jt.grid()
            self.ax_jt.set_xlim(0, plot_length)
            self.ax_jt.set_ylim(min_range, max_range)
            self.ax_jt.set_xlabel("Time (s)", fontsize=12)
            self.ax_jt.set_ylabel(f"{self.joint_name[str(self.joint_index)]} Value(rad)", fontsize=12)
            self.ax_jt.set_title("Joint Values over Time", fontsize=14)

        ### Simulation
        traj = self.sim_joint_traj[:i, self.joint_index]
        self.ax_jt.plot(self.time_steps_jt_2d[:i], traj, linestyle='dashed', label='simulation')
        
        ### Measured
        traj = self.measured_joint_traj[:i, self.joint_index]
        self.ax_jt.plot(self.time_steps_jt_2d[:i], traj, linestyle='dashdot', label='measured')

        self.ax_jt.legend(loc="upper right", fontsize=12)


class AnalyzeMultipleTrajectory(object):
    def __init__(self):
        self.file_names = glob.glob('log/*.npy', recursive=True)
        self.num_files = 1 # int(len(self.file_names) / 4)

        self.sim_ee_traj = []
        self.measured_ee_traj = []
        self.true_ee_traj = []

        self.sim_joint_traj = []
        self.measured_joint_traj = []

        self.ee_traj_len = []
        self.jt_traj_len = []

        for idx in range(self.num_files):
            # Load ee trajectory
            self.sim_ee_traj.append(np.load(f"log/simulation_ee_traj.npy", allow_pickle=True))
            self.measured_ee_traj.append(np.load(f"log/measured_ee_traj.npy", allow_pickle=True))

            # Load joint trajectory
            self.sim_joint_traj.append(np.load(f"log/simulation_joint_traj.npy", allow_pickle=True))
            self.measured_joint_traj.append(np.load(f"log/measured_joint_traj.npy", allow_pickle=True))

            # Length of ee/joint trajectory
            self.ee_traj_len.append(min(len(self.sim_ee_traj[idx]), len(self.measured_ee_traj[idx])))
            self.jt_traj_len.append(min(len(self.sim_joint_traj[idx]), len(self.measured_joint_traj[idx])))

        # Visualization settings
        self.num_joints = 8
        self.joint_name = {'2': 'BaseRotation', '3': 'ArmLift', '4': 'ArmFlex', 
                           '5': 'ArmRoll', '6': 'WristFlex', '7': 'WristRoll'}

    def plot_ee_traj(self, axis='xy'):
        all_in_one = False

        if all_in_one:
            if axis == 'xy':
                fig_xy, ax_xy = plt.subplots()
                ax_xy.set_xlim(0.5, 1.5)
                ax_xy.set_ylim(-1.0, 1.0)
                ax_xy.set_xlabel('X-axis Position (m)')
                ax_xy.set_ylabel('Y-axis Position (m)')
            elif axis == 'yz':
                fig_yz, ax_yz = plt.subplots()
                ax_yz.set_xlim(-1.0, 1.0)
                ax_yz.set_ylim(0.0, 1.0)
                ax_yz.set_xlabel('Y-axis Position (m)')
                ax_yz.set_ylabel('Z-axis Position (m)')
            elif axis == 'zx':
                fig_zx, ax_zx = plt.subplots()
                ax_zx.set_xlim(0.0, 1.0)
                ax_zx.set_ylim(0.5, 1.5)
                ax_zx.set_xlabel('Z-axis Position (m)')
                ax_zx.set_ylabel('X-axis Position (m)')

            for idx in range(self.num_files):
                for i in range(self.ee_traj_len[idx]):
                    sim_traj = self.sim_ee_traj[idx][i]
                    sim_x = sim_traj[0][0]
                    sim_y = sim_traj[0][1]
                    sim_z = sim_traj[0][2]

                    measured_traj = self.measured_ee_traj[idx][i]
                    measured_x = measured_traj[0][0]
                    measured_y = measured_traj[0][1]
                    measured_z = measured_traj[0][2]

                    if axis == 'xy':
                        ax_xy.plot(sim_x, sim_y, marker='*', markersize=3, color='tomato', label='Simulation Trajectory')
                        ax_xy.plot(measured_x, measured_y, marker='^', markersize=3, color='deepskyblue', label='Measured Trajectory')
                    elif axis == 'yz':
                        ax_yz.plot(sim_y, sim_z, marker='*', markersize=3, color='coral', label='Simulation Trajectory')
                        ax_yz.plot(measured_y, measured_z, marker='^', markersize=3, color='goldenrod', label='Measured Trajectory')
                    elif axis == 'zx':
                        ax_zx.plot(sim_z, sim_x, marker='*', markersize=3, color='springgreen', label='Simulation Trajectory')
                        ax_zx.plot(measured_z, measured_x, marker='^', markersize=3, color='lightseagreen', label='Measured Trajectory')

            if axis == 'xy':
                ax_xy.set_title('End Effector XY-axis Trajectory')
                plt.savefig('XY_traj.png')
                plt.close()
            elif axis == 'yz':
                ax_yz.set_title('End Effector YZ-axis Trajectory')
                plt.savefig('YZ_traj.png')
                plt.close()
            elif axis == 'zx':
                ax_zx.set_title('End Effector ZX-axis Trajectory')
                plt.savefig('ZX_traj.png')
                plt.close()

        else:
            for idx in range(self.num_files):
                if axis == 'xy':
                    fig_xy, ax_xy = plt.subplots()
                    ax_xy.set_xlim(0.5, 1.5)
                    ax_xy.set_ylim(-1.0, 1.0)
                    ax_xy.set_xlabel('X-axis Position (m)')
                    ax_xy.set_ylabel('Y-axis Position (m)')
                elif axis == 'yz':
                    fig_yz, ax_yz = plt.subplots()
                    ax_yz.set_xlim(-1.0, 1.0)
                    ax_yz.set_ylim(0.0, 1.0)
                    ax_yz.set_xlabel('Y-axis Position (m)')
                    ax_yz.set_ylabel('Z-axis Position (m)')
                elif axis == 'zx':
                    fig_zx, ax_zx = plt.subplots()
                    ax_zx.set_xlim(0.0, 1.0)
                    ax_zx.set_ylim(0.5, 1.5)
                    ax_zx.set_xlabel('Z-axis Position (m)')
                    ax_zx.set_ylabel('X-axis Position (m)')

                for i in range(self.ee_traj_len[idx]):
                    sim_traj = self.sim_ee_traj[idx][i]
                    sim_x = sim_traj[0][0]
                    sim_y = sim_traj[0][1]
                    sim_z = sim_traj[0][2]

                    measured_traj = self.measured_ee_traj[idx][i]
                    measured_x = measured_traj[0][0]
                    measured_y = measured_traj[0][1]
                    measured_z = measured_traj[0][2]

                    if axis == 'xy':
                        ax_xy.plot(sim_x, sim_y, marker='*', markersize=3, color='tomato', label='Simulation Trajectory')
                        ax_xy.plot(measured_x, measured_y, marker='*', markersize=3, color='deepskyblue', label='Measured Trajectory')
                    elif axis == 'yz':
                        ax_yz.plot(sim_y, sim_z, marker='*', markersize=3, color='coral', label='Simulation Trajectory')
                        ax_yz.plot(measured_y, measured_z, marker='*', markersize=3, color='deepskyblue', label='Measured Trajectory')
                    elif axis == 'zx':
                        ax_zx.plot(sim_z, sim_x, marker='*', markersize=3, color='springgreen', label='Simulation Trajectory')
                        ax_zx.plot(measured_z, measured_x, marker='*', markersize=3, color='deepskyblue', label='Measured Trajectory')

                if axis == 'xy':
                    ax_xy.set_title('End Effector XY-axis Trajectory')
                    plt.savefig('XY_traj.png')
                    plt.close()
                elif axis == 'yz':
                    ax_yz.set_title('End Effector YZ-axis Trajectory')
                    plt.savefig('YZ_traj.png')
                    plt.close()
                elif axis == 'zx':
                    ax_zx.set_title('End Effector ZX-axis Trajectory')
                    plt.savefig('ZX_traj.png')
                    plt.close()

    def evaluate_ee_error_correlation(self, axis='yz'):
        all_in_one = False

        if all_in_one:
            if axis == 'xy':
                fig_xy, ax_xy = plt.subplots()
                ax_xy.set_xlim(-0.5, 0.5)
                ax_xy.set_ylim(-0.5, 0.5)
                ax_xy.set_xlabel('X-axis Error (m)')
                ax_xy.set_ylabel('Y-axis Error (m)')
                x_range = np.arange(-0.5, 0.5, 0.01)
                y_range = np.arange(-0.5, 0.5, 0.01)
                ax_xy.plot(x_range, y_range, color='tan', linestyle="--")
            elif axis == 'yz':
                fig_yz, ax_yz = plt.subplots()
                ax_yz.set_xlim(-0.5, 0.5)
                ax_yz.set_ylim(-0.5, 0.5)
                ax_yz.set_xlabel('Y-axis Error (m)')
                ax_yz.set_ylabel('Z-axis Error (m)')
                x_range = np.arange(-0.5, 0.5, 0.01)
                y_range = np.arange(-0.5, 0.5, 0.01)
                ax_yz.plot(x_range, y_range, color='tan', linestyle="--")
            elif axis == 'zx':
                fig_zx, ax_zx = plt.subplots()
                ax_zx.set_xlim(-0.5, 0.5)
                ax_zx.set_ylim(-0.5, 0.5)
                ax_zx.set_xlabel('Z-axis Error (m)')
                ax_zx.set_ylabel('X-axis Error (m)')
                x_range = np.arange(-0.5, 0.5, 0.01)
                y_range = np.arange(-0.5, 0.5, 0.01)
                ax_zx.plot(x_range, y_range, color='tan', linestyle="--")

            for idx in range(self.num_files):
                diff_l1_all = []
                diff_l1_x = []
                diff_l1_y = []
                diff_l1_z = []
                for i in range(self.ee_traj_len[idx]):
                    sim_traj = self.sim_ee_traj[idx][i]
                    sim_x = sim_traj[0][0]
                    sim_y = sim_traj[0][1]
                    sim_z = sim_traj[0][2]

                    measured_traj = self.measured_ee_traj[idx][i]
                    measured_x = measured_traj[0][0]
                    measured_y = measured_traj[0][1]
                    measured_z = measured_traj[0][2]

                    l1_x = np.array([sim_x - measured_x])
                    l1_y = np.array([sim_y - measured_y])
                    l1_z = np.array([sim_z - measured_z])
                    l1_norm = l1_x + l1_y + l1_z

                    diff_l1_x.append(l1_x)
                    diff_l1_y.append(l1_y)
                    diff_l1_z.append(l1_z)
                    diff_l1_all.append(l1_norm)

                    if axis == 'xy':
                        ax_xy.plot(l1_x, l1_y, marker='x', markersize=3, color='teal', label='XY Error')
                    elif axis == 'yz':
                        ax_yz.plot(l1_y, l1_z, marker='x', markersize=3, color='coral', label='YZ Error')
                    elif axis == 'zx':
                        ax_zx.plot(l1_z, l1_x, marker='x', markersize=3, color='palevioletred', label='ZX Error')

            diff_l1_x = np.array(diff_l1_x)
            diff_l1_y = np.array(diff_l1_y)
            diff_l1_z = np.array(diff_l1_z)

            corr_mat = np.concatenate((diff_l1_x, diff_l1_y, diff_l1_z), axis=1)
            corr_mat = corr_mat.T
            corr_coef = np.corrcoef(corr_mat)
            print('corr_coef:', corr_coef)

            if axis == 'xy':
                coef = corr_coef[0][1]
                ax_xy.set_title(f'End Effector XY-axis Error Correlation: r={coef:.3f}')
                plt.savefig('XY_correlation.png')
                plt.close()
            elif axis == 'yz':
                coef = corr_coef[1][2]
                ax_yz.set_title(f'End Effector YZ-axis Error Correlation: r={coef:.3f}')
                plt.savefig('YZ_correlation.png')
                plt.close()
            elif axis == 'zx':
                coef = corr_coef[2][0]
                ax_zx.set_title(f'End Effector ZX-axis Error Correlation: r={coef:.3f}')
                plt.savefig('ZX_correlation.png')
                plt.close()

        else:
            for idx in range(self.num_files):
                if axis == 'xy':
                    fig_xy, ax_xy = plt.subplots()
                    ax_xy.set_xlim(-0.5, 0.5)
                    ax_xy.set_ylim(-0.5, 0.5)
                    ax_xy.set_xlabel('X-axis Error (m)')
                    ax_xy.set_ylabel('Y-axis Error (m)')
                    x_range = np.arange(-0.5, 0.5, 0.01)
                    y_range = np.arange(-0.5, 0.5, 0.01)
                    ax_xy.plot(x_range, y_range, color='tan', linestyle="--")
                elif axis == 'yz':
                    fig_yz, ax_yz = plt.subplots()
                    ax_yz.set_xlim(-0.5, 0.5)
                    ax_yz.set_ylim(-0.5, 0.5)
                    ax_yz.set_xlabel('Y-axis Error (m)')
                    ax_yz.set_ylabel('Z-axis Error (m)')
                    x_range = np.arange(-0.5, 0.5, 0.01)
                    y_range = np.arange(-0.5, 0.5, 0.01)
                    ax_yz.plot(x_range, y_range, color='tan', linestyle="--")
                elif axis == 'zx':
                    fig_zx, ax_zx = plt.subplots()
                    ax_zx.set_xlim(-0.5, 0.5)
                    ax_zx.set_ylim(-0.5, 0.5)
                    ax_zx.set_xlabel('Z-axis Error (m)')
                    ax_zx.set_ylabel('X-axis Error (m)')
                    x_range = np.arange(-0.5, 0.5, 0.01)
                    y_range = np.arange(-0.5, 0.5, 0.01)
                    ax_zx.plot(x_range, y_range, color='tan', linestyle="--")

                diff_l1_all = []
                diff_l1_x = []
                diff_l1_y = []
                diff_l1_z = []
                for i in range(self.ee_traj_len[idx]):
                    sim_traj = self.sim_ee_traj[idx][i]
                    sim_x = sim_traj[0][0]
                    sim_y = sim_traj[0][1]
                    sim_z = sim_traj[0][2]

                    measured_traj = self.measured_ee_traj[idx][i]
                    measured_x = measured_traj[0][0]
                    measured_y = measured_traj[0][1]
                    measured_z = measured_traj[0][2]

                    l1_x = np.array([sim_x - measured_x])
                    l1_y = np.array([sim_y - measured_y])
                    l1_z = np.array([sim_z - measured_z])
                    l1_norm = l1_x + l1_y + l1_z

                    diff_l1_x.append(l1_x)
                    diff_l1_y.append(l1_y)
                    diff_l1_z.append(l1_z)
                    diff_l1_all.append(l1_norm)

                    if axis == 'xy':
                        ax_xy.plot(l1_x, l1_y, marker='x', markersize=3, color='teal', label='XY Error')
                    elif axis == 'yz':
                        ax_yz.plot(l1_y, l1_z, marker='x', markersize=3, color='coral', label='YZ Error')
                    elif axis == 'zx':
                        ax_zx.plot(l1_z, l1_x, marker='x', markersize=3, color='palevioletred', label='ZX Error')

                diff_l1_x = np.array(diff_l1_x)
                diff_l1_y = np.array(diff_l1_y)
                diff_l1_z = np.array(diff_l1_z)

                corr_mat = np.concatenate((diff_l1_x, diff_l1_y, diff_l1_z), axis=1)
                corr_mat = corr_mat.T
                corr_coef = np.corrcoef(corr_mat)
                print('corr_coef:', corr_coef)

                if axis == 'xy':
                    coef = corr_coef[0][1]
                    ax_xy.set_title(f'End Effector XY-axis Error Correlation: r={coef:.3f}')
                    plt.savefig(f'XY_correlation_{idx}.png')
                    plt.close()
                elif axis == 'yz':
                    coef = corr_coef[1][2]
                    ax_yz.set_title(f'End Effector YZ-axis Error Correlation: r={coef:.3f}')
                    plt.savefig(f'YZ_correlation_{idx}.png')
                    plt.close()
                elif axis == 'zx':
                    coef = corr_coef[2][0]
                    ax_zx.set_title(f'End Effector ZX-axis Error Correlation: r={coef:.3f}')
                    plt.savefig(f'ZX_correlation_{idx}.png')
                    plt.close()

    def evaluate_ee_error(self):
        mean_x_l2_all = []
        mean_y_l2_all = []
        mean_z_l2_all = []
        mean_all_l2_all = []

        std_x_l2_all = []
        std_y_l2_all = []
        std_z_l2_all = []
        std_all_l2_all = []

        for idx in range(self.num_files):
            diff_l2_all = []
            diff_l2_x = []
            diff_l2_y = []
            diff_l2_z = []
            for i in range(self.ee_traj_len[idx]):
                sim_traj = self.sim_ee_traj[idx][i]
                sim_x = sim_traj[0][0]
                sim_y = sim_traj[0][1]
                sim_z = sim_traj[0][2]

                measured_traj = self.measured_ee_traj[idx][i]
                measured_x = measured_traj[0][0]
                measured_y = measured_traj[0][1]
                measured_z = measured_traj[0][2]

                l2_x = np.sqrt(np.square(sim_x-measured_x))
                l2_y = np.sqrt(np.square(sim_y-measured_y))
                l2_z = np.sqrt(np.square(sim_z-measured_z))
                l2_norm = np.sqrt(np.square(sim_x-measured_x)+np.square(sim_y-measured_y)+np.square(sim_z-measured_z))

                diff_l2_x.append(l2_x)
                diff_l2_y.append(l2_y)
                diff_l2_z.append(l2_z)
                diff_l2_all.append(l2_norm)

            # Calculate mean
            mean_l2_x = np.mean(diff_l2_x)
            mean_l2_y = np.mean(diff_l2_y)
            mean_l2_z = np.mean(diff_l2_z)
            mean_l2_all = np.mean(diff_l2_all)
            mean_x_l2_all.append(mean_l2_x)
            mean_y_l2_all.append(mean_l2_y)
            mean_z_l2_all.append(mean_l2_z)
            mean_all_l2_all.append(mean_l2_all)

            # Calculate std
            std_l2_x = np.var(diff_l2_x)
            std_l2_y = np.var(diff_l2_y)
            std_l2_z = np.var(diff_l2_z)
            std_l2_all = np.var(diff_l2_all)
            std_x_l2_all.append(std_l2_x)
            std_y_l2_all.append(std_l2_y)
            std_z_l2_all.append(std_l2_z)
            std_all_l2_all.append(std_l2_all)

            # Calculate max
            max_l2_x = np.max(diff_l2_x)
            max_l2_y = np.max(diff_l2_y)
            max_l2_z = np.max(diff_l2_z)
            max_l2_all = np.max(diff_l2_all)

            print('Mean difference in direction X:', mean_l2_x)
            print('Mean difference in direction Y:', mean_l2_y)
            print('Mean difference in direction Z:', mean_l2_z)
            print('Mean difference: ', mean_l2_all)

            print('Std difference in direction X:', std_l2_x)
            print('Std difference in direction Y:', std_l2_y)
            print('Std difference in direction Z:', std_l2_y)
            print('Std difference:', std_l2_all)

            print('Max difference in direction X:', max_l2_x)
            print('Max difference in direction Y:', max_l2_y)
            print('Max difference in direction Z:', max_l2_z)
            print('Max difference:', max_l2_all)

        ### Visualize mean and std
        all_in_one = True

        # Visualize X direction
        if all_in_one:
            x_range = np.arange(1, self.num_files+1)
            y_range = mean_x_l2_all
            fig, ax = plt.subplots()
            ax.set_title('End-Effector Position Error')
            ax.set_xlim(0.5, self.num_files+0.5)
            ax.set_ylim(-0.05, 0.2)
            ax.set_xlabel('Number of Trials')
            ax.set_ylabel('Mean Error (m)')
            ax.grid()
            ax.errorbar(x_range, y_range, yerr=std_x_l2_all, fmt='-^', markersize=6, color='darkorange', 
                        markeredgecolor='darkorange', ecolor='darkorange', capsize=4, label='X position')

            # Visualize Y direction
            x_range = np.arange(1, self.num_files+1)
            y_range = mean_y_l2_all
            ax.errorbar(x_range, y_range, yerr=std_y_l2_all, fmt='-d', markersize=6, color='chocolate', 
                        markeredgecolor='chocolate', ecolor='chocolate', capsize=4, label='Y position')

            # Visualize Z direction
            x_range = np.arange(1, self.num_files+1)
            y_range = mean_z_l2_all
            ax.errorbar(x_range, y_range, yerr=std_z_l2_all, fmt='-o', markersize=6, color='olivedrab', 
                        markeredgecolor='olivedrab', ecolor='olivedrab', capsize=4, label='Z position')

            # Visualize All
            x_range = np.arange(1, self.num_files+1)
            y_range = mean_all_l2_all
            ax.errorbar(x_range, y_range, yerr=std_all_l2_all, fmt='-*', markersize=6, color='midnightblue', 
                        markeredgecolor='midnightblue', ecolor='midnightblue', capsize=4, label='XYZ position')
    
            ax.legend(loc="upper right")
        else:
            x_range = np.arange(1, self.num_files+1)
            y_range = mean_x_l2_all
            fig, ax = plt.subplots()
            ax.set_title('End Effector Mean Error in X Direction')
            ax.set_xlim(0.5, self.num_files+0.5)
            ax.set_ylim(-0.05, 0.2)
            ax.set_xlabel('Number of Trials')
            ax.set_ylabel('Mean Error (m)')
            ax.grid()

            ax.errorbar(x_range, y_range, yerr=std_x_l2_all, fmt='-^', markersize=6, color='darkorange', 
                        markeredgecolor='darkorange', ecolor='darkorange', capsize=4)
            plt.show()
            plt.close()

            # Visualize Y direction
            x_range = np.arange(1, self.num_files+1)
            y_range = mean_y_l2_all
            fig, ax = plt.subplots()
            ax.set_title('End Effector Mean Error in Y Direction')
            ax.set_xlim(0.5, self.num_files+0.5)
            ax.set_ylim(-0.05, 0.2)
            ax.set_xlabel('Number of Trials')
            ax.set_ylabel('Mean Error (m)')
            ax.grid()

            ax.errorbar(x_range, y_range, yerr=std_y_l2_all, fmt='-^', markersize=6, color='darkorange', 
                        markeredgecolor='darkorange', ecolor='darkorange', capsize=4)

            # Visualize Z direction
            x_range = np.arange(1, self.num_files+1)
            y_range = mean_z_l2_all
            fig, ax = plt.subplots()
            ax.set_title('End Effector Mean Error in Z Direction')
            ax.set_xlim(0.5, self.num_files+0.5)
            ax.set_ylim(-0.05, 0.2)
            ax.set_xlabel('Number of Trials')
            ax.set_ylabel('Mean Error (m)')
            ax.grid()

            ax.errorbar(x_range, y_range, yerr=std_z_l2_all, fmt='-^', markersize=6, color='darkorange', 
                        markeredgecolor='darkorange', ecolor='darkorange', capsize=4)
            plt.show()
            plt.close()

            # Visualize All
            x_range = np.arange(1, self.num_files+1)
            y_range = mean_all_l2_all
            fig, ax = plt.subplots()
            ax.set_title('End Effector Mean Error')
            ax.set_xlim(0.5, self.num_files+0.5)
            ax.set_ylim(-0.05, 0.2)
            ax.set_xlabel('Number of Trials')
            ax.set_ylabel('Mean Error (m)')
            ax.grid()

            ax.errorbar(x_range, y_range, yerr=std_all_l2_all, fmt='-^', markersize=6, color='darkorange', 
                        markeredgecolor='darkorange', ecolor='darkorange', capsize=4)
            plt.show()
            plt.savefig('mea_error.png')

    def evaluate_jt_error(self):
        mean_joint_all = []
        std_joint_all = []
        max_joint_all = []
        for idx in range(self.num_files):
            mean_joint_each = []
            std_joint_each = []
            max_joint_each = []
            for i in range(self.num_joints):
                diff_each_joints = []
                for j in range(self.jt_traj_len[idx]):
                    sim_joint = self.sim_joint_traj[idx][j, i]
                    measured_joint = self.measured_joint_traj[idx][j, i]

                    diff_joint = np.sqrt(np.square(sim_joint-measured_joint))
                    diff_each_joints.append(diff_joint)
                
                mean_joint = np.mean(diff_each_joints)
                std_joint = np.var(diff_each_joints)
                max_joint = np.max(diff_each_joints)

                print(f"Mean joint_{i} difference:", mean_joint)
                print(f"Std joint_{i} difference:", std_joint)
                print(f"Max joint_{i} difference:", max_joint)

                mean_joint_each.append(mean_joint)
                std_joint_each.append(std_joint)
                max_joint_each.append(max_joint)

            mean_joint_all.append(mean_joint_each)
            std_joint_all.append(std_joint_each)
            max_joint_all.append(max_joint_all)

        # Visualize results
        for i in range(self.num_joints):
            if i == 0:
                x_range = np.arange(1, self.num_files+1)
                y_range = np.array(mean_joint_all).T[i]

                fig, ax = plt.subplots()
                ax.set_title('X Position Error')
                ax.set_xlim(0.5, self.num_files+0.5)
                ax.set_ylim(-0.05, 0.15)
                ax.set_xlabel('Number of Trials')
                ax.set_ylabel('Position Error (m)')
                ax.grid()

                ax.errorbar(x_range, y_range, yerr=np.array(std_joint_all).T[i], fmt='-^', markersize=6, color='navy', 
                            markeredgecolor='navy', ecolor='navy', capsize=4, label='X Position Error')
                ax.legend(loc="upper right")
                plt.savefig('joint_x_error.png')
                plt.close()

            elif i == 1:
                x_range = np.arange(1, self.num_files+1)
                y_range = np.array(mean_joint_all).T[i]

                fig, ax = plt.subplots()
                ax.set_title('Y Position Error')
                ax.set_xlim(0.5, self.num_files+0.5)
                ax.set_ylim(-0.05, 0.15)
                ax.set_xlabel('Number of Trials')
                ax.set_ylabel('Position Error (m)')
                ax.grid()

                ax.errorbar(x_range, y_range, yerr=np.array(std_joint_all).T[i], fmt='-^', markersize=6, color='navy', 
                            markeredgecolor='navy', ecolor='navy', capsize=4, label='Y Position Error')
                ax.legend(loc="upper right")
                plt.savefig('joint_y_error.png')
                plt.close()

            elif i == 3:
                x_range = np.arange(1, self.num_files+1)
                y_range = np.array(mean_joint_all).T[i]

                fig, ax = plt.subplots()
                ax.set_title('ArmLift Joint Error')
                ax.set_xlim(0.5, self.num_files+0.5)
                ax.set_ylim(-0.05, 0.15)
                ax.set_xlabel('Number of Trials')
                ax.set_ylabel('Position Error (m)')
                ax.grid()

                ax.errorbar(x_range, y_range, yerr=np.array(std_joint_all).T[i], fmt='-^', markersize=6, color='navy', 
                            markeredgecolor='navy', ecolor='navy', capsize=4, label='ArmLift Position Error')
                ax.legend(loc="upper right")
                plt.savefig('joint_armlift_error.png')
                plt.close()

            else:
                x_range = np.arange(1, self.num_files+1)
                y_range = np.array(mean_joint_all).T[i]

                fig, ax = plt.subplots()
                ax.set_title(f'{self.joint_name[str(i)]} Position Error')
                ax.set_xlim(0.5, self.num_files+0.5)
                ax.set_ylim(-0.05, 0.15)
                ax.set_xlabel('Number of Trials')
                ax.set_ylabel('Position Error (rad)')
                ax.grid()

                ax.errorbar(x_range, y_range, yerr=np.array(std_joint_all).T[i], fmt='-^', markersize=6, color='navy', 
                            markeredgecolor='navy', ecolor='navy', capsize=4, label=f'{self.joint_name[str(i)]} Position Error')
                ax.legend(loc="upper right")
                plt.savefig(f'joint_{self.joint_name[str(i)].lower()}_error.png')
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Visualize trajectory')
    parser.add_argument('--index', type=int, default=0, help='number of files for visualization')
    parser.add_argument('--joint_index', type=int, default=None, help='set joint index')
    parser.add_argument('--mode', type=str, default='both', help='set visualization mode')
    parser.add_argument('--visualize', action='store_true', help='set visualization')
    args = parser.parse_args()

    if args.visualize:
        at = AnalyzeTrajectory(mode=args.mode, joint_index=args.joint_index)
        at.visualize_ee_3d_traj()
        at.visualize_joint_2d_traj()
    else:
        at = AnalyzeMultipleTrajectory()
        at.evaluate_ee_error()
        at.evaluate_jt_error()
        at.evaluate_ee_error_correlation()
        at.plot_ee_traj()