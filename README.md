# TAMP-HSR
This repository implement PDDLStream for Toyota Human Support Robot (HSR) and offer parallel reinforcement learning environment on Isaac Sim.

## Getting Started
### Prerequisited
- NVIDIA Docker
- NVIDIA RTX GPU
- NVIDIA Driver 515.xx

https://github.com/makolon/hsr_isaac_tamp/assets/39409362/e7945ca0-e040-47cc-b73f-0cf99413d30d

https://github.com/makolon/hsr_isaac_tamp/assets/39409362/0322855f-2aa6-46a2-963e-28bc1f77577c

https://github.com/makolon/hsr_isaac_tamp/assets/39409362/e42012fd-bc8a-4455-a068-cca886e32f79

https://github.com/makolon/hsr_isaac_tamp/assets/39409362/c88e78c7-af53-4cb1-a4f9-ba2a95cc8640

https://github.com/makolon/hsr_isaac_tamp/assets/39409362/734c6f8a-b9e4-45f5-a454-3dce7bddddb7

### Installation

1. Clone the repository
```
$ git clone --recursive git@github.com/makolon/tamp-hsr.git
```

2. Build docker image
```
$ cd tamp-hsr/docker/docker_hsr/
$ ./build.sh
```

3. Run docker container
```
$ cd tamp-hsr/docker/docker_hsr/
$ ./run.sh
```

4. Compile FastDownward
```
$ cd tamp-hsr/hsr_tamp/downward/
$ git submodule update --init --recursive
$ python3 build.py
$ cd ./builds/
$ ln -s release release32
```

## Usage
### Simulation
#### PDDLStream only for 2d environment
You can test PDDLStream on 2D pygame environment.
```
$ cd tamp-hsr/hsr_tamp/experiments/env_2d/
$ python3 tamp_planner.py
```

#### PDDLStream only for 3d environment
You can test PDDLStream on 3D pybullet environment including cooking, holding block task.
```
$ cd tamp-hsr/hsr_tamp/experiments/env_3d/
$ python3 tamp_planner.py --problem <problem_name>
```

### Real
#### Execute plan on 2D environment
1. Enter to hsrb_mode
```
$ hsrb_mode
```

2. Set up ROS configurations
```
$ cd tamp-hsr/hsr_ros/hsr_ws/
$ source devel/setup.bash
```

3. Execute ROS scripts
```
$ roslaunch env_2d exec_tamp.launch --mode <feedforward/feedback>
```

#### Execute plan on 3D environment
1. Enter to hsrb_mode
```
$ hsrb_mode
```

2. Set up ROS configurations
```
$ cd tamp-hsr/hsr_ros/hsr_ws/
$ source devel/setup.bash
```

3. Launch gearbox assembling scipts
```
$ roslaunch env_3d exec_tamp.launch --mode <feedforward/feedback>
````

## Setup IKfast
### Compile IKfast
Build & run docker for openrave that contain IKfast scripts.
```
$ cd tamp-hsr/docker/docker_openrave/
$ ./build.sh
$ ./run.sh
```
Then, execute ikfast scripts that can automatically create cpp IK solver and copy and plaste the appropriate scripts to <ik_solver>.cpp.
```
$ ./exec_openrave.sh
```
After that process, you can call IK solver in python script by executing the following commands.
```
$ cd tamp-hsr/hsr_tamp/experiments/env_3d/utils/pybullet_tools/ikfast/hsrb/
$ python3 setup.py
```

### Create HSR Collada Model
If you don't have hsr collada model, you have to run the following commands in docker_openrave container. \
Terminal 1.
```
$ cd /ikfast/
$ roscore
```
Terminal 2.
```
$ cd /ikfast
$ export MYROBOT_NAME='hsrb4s'
$ rosrun collada_urdf urdf_to_collada "$MYROBOT_NAME".urdf "$MYROBOT_NAME".dae
```
Then, you can see the generated HSR collada model using following commands.
```
$ openrave-robot.py "$MYROBOT_NAME".dae --info links
$ openrave "$MYROBOT_NAME".dae
```

For more informations, please refer to the [following document](http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/ikfast/ikfast_tutorial.html).

## Setup Motion Capture
### Prerequisited
- Windows11 PC
- Windows11 with ROS
- OptiTrack

### Usage
Check IP address of the HSR PC and own IP address.
```
ipconfig
```
Then, execute the following commands in the ROS executable terminal.
```
set ROS_HOSTNAME=<OWN IP ADDRESS>
set ROS_IP=<OWN IP ADDRESS>
set ROS_MASTER_URI=http://<HSR PC IP ADDRESS>:11311
C:\path\to\catkin_ws\devel\setup.bat
cd C:\path\to\catkin_ws\src\mocap_msg\src && python MocapRosPublisher.py
```
