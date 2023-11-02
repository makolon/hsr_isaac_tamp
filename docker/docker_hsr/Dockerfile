FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

WORKDIR /root/

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        iproute2 \
        g++-7 \
        gcc-7 \
        wget \
        vim \
        git \
        curl \
        gnupg \
        xvfb \
        terminator \
        build-essential \
        openssh-client \
        net-tools \
        lsb-release \
        tk-dev \
        xorg-dev \
        software-properties-common \
        libglu1-mesa-dev \
        libsdl2-dev \
        libc++-7-dev \
        libc++abi-7-dev \
        ninja-build \
        libxi-dev \
        libtbb-dev \
        libosmesa6-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-get install -y build-essential
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

### Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        python3-catkin-tools \
        python3-pyqt5 \
        python3-pip \
        python3-catkin-pkg \
        python3-catkin-tools \
        python3-catkin-pkg-modules \
        python3-rospkg-modules \
        python3-opencv \
        python3-empy \
        python3-dev \
        python3-tk \
        ros-noetic-tf* \
        ros-noetic-ros-numpy* \
    && rm -rf /var/lib/apt/lists/*

### Install python packages
RUN pip3 install -U pip
RUN pip3 install \
    setuptools==45.2.0 \
	gym==0.26.2 \
	pybullet==3.2.5 \
	matplotlib==3.1.2 \
	opencv-python==4.8.0.76 \
	scipy==1.10.1 \
	scikit-image==0.21.0 \
	transforms3d==0.4.1 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    kinpy==0.0.5 \
    pin==2.6.20 \
    trimesh==3.23.5 \
    pygame==2.5.1 \
    catkin_tools==0.9.2 \
    rl-games==1.5.2 \
    warp-lang==0.10.1 \
    pytorch3d==0.3.0 \
    hydra-core==1.3.2 \
	tensorflow==2.13.0 \
	tensorflow-addons==0.21.0 \
	tensorflow_hub==0.14.0 \
	keras==2.13.1 \
	jupyterlab==4.0.5

### Install ROS packages
RUN mkdir -p /root/catkin_ws/src \
    && cd /root/catkin_ws/ \
    && rosdep init \
    && rosdep update

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-noetic-joint-state-publisher-gui \
        ros-noetic-ros-control \
        ros-noetic-ros-controllers \
        ros-noetic-gazebo-ros-control \
        ros-noetic-moveit \
        ros-noetic-moveit-ros-visualization \
        ros-noetic-mocap-optitrack \
    && rm -rf /var/lib/apt/lists/*

### System settings
ENV ROS_PACKAGE_PATH=/root/catkin_ws:$ROS_PACKAGE_PATH
ENV ROS_WORKSPACE=/root/catkin_ws
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "export DISPLAY=:0" >> ~/.bashrc
RUN echo "export PYTHONPATH='/root/tamp-hsr/hsr_tamp:$PYTHONPATH'" >> ~/.bashrc

### Install CUDA
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb && rm cuda-keyring_1.0-1_all.deb

### Install HSR packages
RUN sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/ros/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/tmc.list'
RUN sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/tmc/ubuntu `lsb_release -cs` multiverse main" >> /etc/apt/sources.list.d/tmc.list'
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget https://hsr-user:jD3k4G2e@packages.hsr.io/tmc.key -O - | sudo apt-key add -
RUN wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -O - | sudo apt-key add -
RUN wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
RUN sh -c 'mkdir -p /etc/apt/auth.conf.d'
RUN sh -c '/bin/echo -e "machine packages.hsr.io\nlogin hsr-user\npassword jD3k4G2e" > /etc/apt/auth.conf.d/auth.conf'
RUN apt-get update && apt-get install -y ros-noetic-tmc-desktop-full
COPY ./hsrrc /root/
RUN cat /root/hsrrc >> ~/.bashrc

### Terminator window settings
COPY config /

### User group settings
COPY entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"] /
