#!/bin/bash

docker run -it \
  -v ~/tamp-hsr:/root/tamp-hsr \
  -v ~/docker/isaac-sim/kit/cache/Kit:/isaac-sim/kit/cache/Kit:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -e ACCEPT_EULA=Y \
  -e DISPLAY=unix$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --net=host \
  --gpus all --rm --name "hsr_isaac" hsr_isaac:2022.2.1
