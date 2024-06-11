FROM nvcr.io/nvidia/isaac-sim:2022.2.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    gdb \
    vim \
    htop \
    wget \
    sudo \
    curl \
    cmake \
    gedit \
    unzip \
    xterm \
    gnupg2 \
    locales \
    valgrind \
    dbus-x11 \
    net-tools \
    terminator \
    subversion \
    libssl-dev \
    lsb-release \
    python3-pip \
    bash-completion \
    build-essential \
    software-properties-common && \
    apt-get clean && rm -rf /var/lib/apt/list*

# Setup Isaac Sim
ENV ISAACSIM_PATH="${HOME}/isaac-sim"
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

RUN echo "alias PYTHON_PATH=/isaac-sim/python.sh" >> ${HOME}/.bashrc
RUN echo "export DISPLAY=:1" >> ${HOME}/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:/root/tamp-hsr/" >> ${HOME}/.bashrc

# Setup Python Package
RUN apt-get update
RUN ${ISAACSIM_PYTHON_EXE} -m pip install -U pip
RUN ${ISAACSIM_PYTHON_EXE} -m pip install \
    numpy \
    scipy \
    pybullet \
    setuptools \
    jupyterlab \
    torch \
    torchvision \
    pytorch3d \
    d3rlpy \
    absl-py==1.2.0 \
    aiosignal==1.2.0 \
    antlr4-python3-runtime==4.8 \
    cachetools==4.2.4 \
    cloudpickle==2.1.0 \
    cycler==0.11.0 \
    cython==0.29.32 \
    decorator==4.4.2 \
    distlib==0.3.5 \
    docker-pycreds==0.4.0 \
    eigenpy==2.6.9 \
    filelock==3.7.1 \
    fonttools==4.34.4 \
    frozenlist==1.3.0 \
    gitdb==4.0.9 \
    gitpython==3.1.27 \
    google-auth==1.35.0 \
    google-auth-oauthlib==0.4.6 \
    grpcio==1.43.0 \
    hpp-fcl==1.7.8 \
    hydra-core==1.1.1 \
    imageio==2.21.1 \
    imageio-ffmpeg==0.4.7 \
    importlib-metadata==4.12.0 \
    importlib-resources==5.9.0 \
    iniconfig==1.1.1 \
    joblib==1.1.0 \
    kiwisolver==1.4.4 \
    markdown==3.4.1 \
    matplotlib==3.5.2 \
    moviepy==1.0.3 \
    msgpack==1.0.4 \
    numpy-ml==0.1.2 \
    oauthlib==3.2.0 \
    omegaconf==2.1.1 \
    opencv-python==4.6.0.66 \
    packaging==21.3 \
    pandas==1.3.5 \
    pin==2.6.0 \
    platformdirs==2.5.2 \
    pluggy==1.0.0 \
    proglog==0.1.10 \
    promise==2.3 \
    protobuf==3.20.1 \
    py==1.11.0 \
    pyasn1==0.4.8 \
    pyasn1-modules==0.2.8 \
    pygame==2.1.2 \
    pyparsing==3.0.9 \
    pysdf==0.1.9 \
    pytest==7.1.2 \
    python-dateutil==2.8.2 \
    pytz==2022.1 \
    qpsolvers==2.2.0 \
    quadprog==0.1.11 \
    ray==1.13.0 \
    requests-oauthlib==1.3.1 \
    rl-games==1.5.2 \
    rsa==4.9 \
    scikit-learn==1.0.2 \
    sentry-sdk==1.9.5 \
    setproctitle==1.2.3 \
    shapely==1.8.5 \
    shortuuid==1.0.9 \
    smmap==5.0.0 \
    stable-baselines3==1.2.0 \
    tensorboard==2.4.0 \
    tensorboard-plugin-wit==1.7.0 \
    tensorboardx==2.5.1 \
    threadpoolctl==3.1.0 \
    tomli==2.0.1 \
    tqdm==4.64.0 \
    trimesh==4.0.1 \
    urdfpy==0.0.22 \
    virtualenv==20.15.1 \
    wandb==0.13.1 \
    werkzeug==2.1.2 \
    zipp==3.8.1

# Setup Workdir
WORKDIR /root

SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["/bin/bash"]

