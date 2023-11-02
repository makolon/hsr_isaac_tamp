FROM personalrobotics/ros-openrave

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update || true && apt-get install -y --no-install-recommends \
        vim \
        build-essential \
        python-pip \
        ca-certificates \
        liblapack-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update || true && apt-get install -y --no-install-recommends \
    ros-indigo-collada-* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/sympy/sympy.git@sympy-0.7.1

RUN echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc
RUN echo "source /ikfast/entrypoint.sh" >> ~/.bashrc
RUN echo "export DISPLAY=:1" >> ~/.bashrc
