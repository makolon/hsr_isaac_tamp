#!/usr/bin/env python
# vim: fileencoding=utf-8 :
# Copyright (c) 2016 TOYOTA MOTOR CORPORATION
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#  * Neither the name of Toyota Motor Corporation nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import glob
import os
import subprocess
import tempfile

from nose.tools import eq_, ok_


try:
    import xml.etree.cElementTree as etree
except Exception:
    import xml.etree.ElementTree as etree


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOTS_DIR = os.path.join(PACKAGE_DIR, "robots")
URDF_DIR = os.path.join(PACKAGE_DIR, "urdf")


def test_generator_robot_urdf():
    def test_robot_urdf(path):
        u"""XACROで変換した後に、URDFとして正しく読めるかを確認するテスト"""
        with tempfile.NamedTemporaryFile() as f:
            args = ['rosrun', 'xacro', 'xacro', '--inorder', source]
            eq_(subprocess.call(args, stdout=f), 0)
            args = ['check_urdf', f.name]
            subprocess.check_output(args)

    matched = glob.glob(ROBOTS_DIR + "/*.urdf.xacro")
    sources = [os.path.abspath(path) for path in matched]
    for source in sources:
        yield test_robot_urdf, source


def test_generator_integrity():
    def check_integrity(source):
        args = ['rosrun', 'xacro', 'xacro', '--inorder', source]
        urdf = subprocess.check_output(args)
        root = etree.fromstring(urdf)

        links = []
        for link in root.findall('link'):
            name = link.get('name')
            ok_(name is not None, "link({0})".format(name))
            links.append(name)

        joints = []
        for joint in root.findall('joint'):
            name = joint.get('name')
            ok_(name is not None, "joint({0})".format(name))
            joints.append(name)
            parent = joint.find('parent')
            ok_(parent.get('link') in links, "joint({0})".format(name))
            child = joint.find('child')
            ok_(child.get('link') in links, "joint({0})".format(name))

        for trans in root.findall('transmission'):
            name = trans.get('name')
            joint = trans.find('joint')
            ok_(joint.get('name') in joints, "transmission({0})".format(name))

        for gazebo in root.findall('gazebo'):
            ref = gazebo.get('reference')
            if ref is None:
                # When reference is None, <gazebo> tag is added to <robot>.
                continue
            ok_(ref in links + joints,
                "Unresolvable reference '{0}':\n{1}".format(
                    ref, etree.tostring(gazebo)))

    matched = glob.glob(ROBOTS_DIR + "/*.urdf.xacro")
    sources = [os.path.abspath(path) for path in matched]
    for source in sources:
        yield check_integrity, source
