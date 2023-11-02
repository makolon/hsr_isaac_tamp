#!/bin/bash
docker run -it --rm --privileged --gpus all --net host --ipc host \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ~/tamp-hsr:/root/tamp-hsr \
    --name hsr_tamp_ros hsr_tamp_ros:latest bash