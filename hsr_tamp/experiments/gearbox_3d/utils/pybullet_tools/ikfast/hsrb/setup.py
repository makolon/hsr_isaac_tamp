#!/usr/bin/env python

from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from pybullet_tools.ikfast.compile import compile_ikfast

# Build C++ extension by running: 'python setup.py'
# see: https://docs.python.org/3/extending/building.html

ARMS = ['arm']

def main():
    sys.argv[:] = sys.argv[:1] + ['build']
    compile_ikfast(module_name='ikArm', cpp_filename='arm_ik.cpp')

if __name__ == '__main__':
    main()
