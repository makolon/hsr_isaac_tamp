Overview
++++++++

This package manages the HSR-B robot model.
Gazebo simulator configurations are also included.。


Folder structure
+++++++++++++++++

- **launch**

     Contains a launch file for performing confirmation (visualization) of a robot model.

- **urdf**

    The xacro files that include common macros are directly under this directory.

    Each part is separated into folders named ${PARTNAME}_${VERSION}.

    When a major change such as a model revision is caused by hardware repairs, the version is increased by one and a new folder is created.

    Basically, the version is attached to the part name and is of the form of v0, v1, v2, etc.

- **robots**

    The URDF description file of the entire robot configured using the URDF of each part is placed in this directory.

    There are multiple robot models depending on the version of various hardware parts.

xacro file variables
+++++++++++++++++++++

- **personal_name**

    A namespace to set to allow one gazebo world to display several HSRs.

　　　　personal_name contains the value that is set as an argument in namespace when the launch file is started. The default value is "".

- **robot_name**

    The model name used as the topic name or the service name of the robot. The default is "hsrb".

　　The model name assigned here and the model name actually used are not necessarily the same.

LICENSE
++++++++++++

This software is released under the BSD 3-Clause Clear License, see LICENSE.txt.
