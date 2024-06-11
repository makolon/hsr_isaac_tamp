#Copyright © 2018 Naturalpoint
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# OptiTrack NatNet direct depacketization library for Python 3.x

import os
import tf
import sys
import math
import time
import signal
import socket
import struct
import threading

import rospy
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Vector3,
    Quaternion,
)

def trace( *args ):
    pass

def quaternion_to_euler(quaternion):
    """Convert Quaternion to Euler Angles

    quarternion: geometry_msgs/Quaternion
    euler: geometry_msgs/Vector3
    """
    e = tf.transformations.euler_from_quaternion( (quaternion.x, quaternion.y, quaternion.z, quaternion.w) )
    return list(e)

def euler_to_quaternion(euler):
    """Convert Euler Angles to Quaternion

    euler: geometry_msgs/Vector3
    quaternion: geometry_msgs/Quaternion
    """
    q = tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
    return list(q)

# Create structs for reading various object types to speed up parsing.
Vector3 = struct.Struct( '<fff' )
Quaternion = struct.Struct( '<ffff' )
FloatValue = struct.Struct( '<f' )
DoubleValue = struct.Struct( '<d' )

class NatNetClient:
    def __init__( self ):
        # Change this value to the IP address of the NatNet server.
        self.serverIPAddress = "127.0.0.1" 

        # Change this value to the IP address of your local network interface
        self.localIPAddress = "127.0.0.1"

        # This should match the multicast address listed in Motive's streaming settings.
        self.multicastAddress = "239.255.42.99"

        # NatNet Command channel
        self.commandPort = 1510
        
        # NatNet Data channel     
        self.dataPort = 1511

        # NatNet stream version. This will be updated to the actual version the server is using during initialization.
        self.__natNetStreamVersion = (3,0,0,0)

        rigid_body_list = (
            "base",
            "blue_gear",
            "green_gear",
            "red_gear",
            "yellow_shaft",
        )

        print("ROS node initializing ...")
        self.pub_list = [
            rospy.Publisher('mocap_pose_topic/{0}_pose'.format(rigid_body_list[i]), PoseStamped, queue_size=1)
            for i in range(len(rigid_body_list))
        ]
        rospy.init_node('mocap_pose_node_pub', anonymous=True)
        self.r = rospy.Rate(150)  # [Hz]
        print("ROS node initialized")

    # Client/server message ids
    NAT_PING                  = 0 
    NAT_PINGRESPONSE          = 1
    NAT_REQUEST               = 2
    NAT_RESPONSE              = 3
    NAT_REQUEST_MODELDEF      = 4
    NAT_MODELDEF              = 5
    NAT_REQUEST_FRAMEOFDATA   = 6
    NAT_FRAMEOFDATA           = 7
    NAT_MESSAGESTRING         = 8
    NAT_DISCONNECT            = 9 
    NAT_UNRECOGNIZED_REQUEST  = 100

    # This is a callback function that gets connected to the NatNet client and called once per mocap frame.
    def newFrameListener( self, frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                        labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged ):
        trace( "Received frame", frameNumber )

    # This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
    def rigidBodyListener( self, id, position, rotation, trackingValidFlag ):
        if not trackingValidFlag:
            return -1

        trace( "Received frame for rigid body:" )
        trace( "\tid:{0}, \n\tposition:{1}, \n\trotation*{2}".format(id, position, rotation) )

        """
        # making tf
        br = tf.TransformBroadcaster()
        # mocap world tf
        br.sendTransform(
            (position[0], position[1], position[2]),
            (rotation[0], rotation[1], rotation[2], rotation[3]),
            rospy.Time.now(),
            self.pub_list[id-1].name[17:],
            # "mocap_world")
            "world")
        """

        rigid_pose = Pose()

        rigid_pose.position.x = position[0]
        rigid_pose.position.y = position[1]
        rigid_pose.position.z = position[2]
        
        rigid_pose.orientation.x = rotation[0]
        rigid_pose.orientation.y = rotation[1]
        rigid_pose.orientation.z = rotation[2]
        rigid_pose.orientation.w = rotation[3]

        rigid_posestamped = PoseStamped()
        rigid_posestamped.pose = rigid_pose
        rigid_posestamped.header.stamp = rospy.Time.now()
        rigid_posestamped.header.frame_id = "world"

        self.pub_list[id-1].publish(rigid_posestamped)

    # Create a data socket to attach to the NatNet stream
    def __createDataSocket( self, port ):
        result = socket.socket( socket.AF_INET, # Internet
                              socket.SOCK_DGRAM,
                              socket.IPPROTO_UDP) # UDP

        result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)        
        result.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.multicastAddress) + socket.inet_aton(self.localIPAddress))

        result.bind( (self.localIPAddress, port) )

        return result

    # Create a command socket to attach to the NatNet stream
    def __createCommandSocket( self ):
        result = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )
        result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        result.bind( ('', 0) )
        result.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        return result

    # Unpack a rigid body object from a data packet
    def __unpackRigidBody( self, data ):
        offset = 0

        # ID (4 bytes)
        id = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4
        trace( "ID:", id )

        # Position and orientation
        pos = Vector3.unpack( data[offset:offset+12] )
        offset += 12
        trace( "\tPosition:", pos[0],",", pos[1],",", pos[2] )
        rot = Quaternion.unpack( data[offset:offset+16] )
        offset += 16
        trace( "\tOrientation:", rot[0],",", rot[1],",", rot[2],",", rot[3] )

        # RB Marker Data ( Before version 3.0.  After Version 3.0 Marker data is in description )
        if( self.__natNetStreamVersion[0] < 3  and self.__natNetStreamVersion[0] != 0) :
            # Marker count (4 bytes)
            markerCount = struct.unpack('i',  data[offset:offset+4])[0]
            offset += 4
            markerCountRange = range( 0, markerCount )
            trace( "\tMarker Count:", markerCount )

            # Marker positions
            for i in markerCountRange:
                pos = Vector3.unpack( data[offset:offset+12] )
                offset += 12
                trace( "\tMarker", i, ":", pos[0],",", pos[1],",", pos[2] )

            if( self.__natNetStreamVersion[0] >= 2 ):
                # Marker ID's
                for i in markerCountRange:
                    id = struct.unpack('i',  data[offset:offset+4])[0]
                    offset += 4
                    trace( "\tMarker ID", i, ":", id )

                # Marker sizes
                for i in markerCountRange:
                    size = FloatValue.unpack( data[offset:offset+4] )
                    offset += 4
                    trace( "\tMarker Size", i, ":", size[0] )
                    
        if( self.__natNetStreamVersion[0] >= 2 ):
            markerError, = FloatValue.unpack( data[offset:offset+4] )
            offset += 4
            trace( "\tMarker Error:", markerError )

        # Version 2.6 and later
        if( ( ( self.__natNetStreamVersion[0] == 2 ) and ( self.__natNetStreamVersion[1] >= 6 ) ) or self.__natNetStreamVersion[0] > 2 or self.__natNetStreamVersion[0] == 0 ):
            param, = struct.unpack( 'h', data[offset:offset+2] )
            trackingValid = ( param & 0x01 ) != 0
            offset += 2
            trackingValidFlag = True if trackingValid else False
            trace( "\tTracking Valid:", trackingValidFlag )

        self.rigidBodyListener( id, pos, rot, trackingValidFlag )

        return offset

    # Unpack a skeleton object from a data packet
    def __unpackSkeleton( self, data ):
        offset = 0
        
        id = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4
        trace( "ID:", id )
        
        rigidBodyCount = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4
        trace( "Rigid Body Count:", rigidBodyCount )
        for j in range( 0, rigidBodyCount ):
            offset += self.__unpackRigidBody( data[offset:] )

        return offset

    # Unpack data from a motion capture frame message
    def __unpackMocapData( self, data ):
        trace( "Begin MoCap Frame\n-----------------\n" )

        # data = memoryview( data )
        offset = 0
        
        # Frame number (4 bytes)
        frameNumber = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4
        trace( "Frame #:", frameNumber )

        # Marker set count (4 bytes)
        markerSetCount = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4
        trace( "Marker Set Count:", markerSetCount )

        for i in range( 0, markerSetCount ):
            # Model name
            modelName, separator, remainder = bytes(data[offset:]).partition( b'\0' )
            offset += len( modelName ) + 1
            trace( "Model Name:", modelName.decode( 'utf-8' ) )

            # Marker count (4 bytes)
            markerCount = struct.unpack('i',  data[offset:offset+4])[0]
            offset += 4
            trace( "Marker Count:", markerCount )

            for j in range( 0, markerCount ):
                pos = Vector3.unpack( data[offset:offset+12] )
                offset += 12
                #trace( "\tMarker", j, ":", pos[0],",", pos[1],",", pos[2] )
                 
        # Unlabeled markers count (4 bytes)
        unlabeledMarkersCount = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4
        trace( "Unlabeled Markers Count:", unlabeledMarkersCount )

        for i in range( 0, unlabeledMarkersCount ):
            pos = Vector3.unpack( data[offset:offset+12] )
            offset += 12
            trace( "\tMarker", i, ":", pos[0],",", pos[1],",", pos[2] )

        # Rigid body count (4 bytes)
        rigidBodyCount = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4
        trace( "Rigid Body Count:", rigidBodyCount )

        for i in range( 0, rigidBodyCount ):
            offset += self.__unpackRigidBody( data[offset:] )

        # Version 2.1 and later
        skeletonCount = 0
        if( ( self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] > 0 ) or self.__natNetStreamVersion[0] > 2 ):
            skeletonCount = struct.unpack('i',  data[offset:offset+4])[0]
            offset += 4
            trace( "Skeleton Count:", skeletonCount )
            for i in range( 0, skeletonCount ):
                offset += self.__unpackSkeleton( data[offset:] )

        # Labeled markers (Version 2.3 and later)
        labeledMarkerCount = 0
        if( ( self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] > 3 ) or self.__natNetStreamVersion[0] > 2 ):
            labeledMarkerCount = struct.unpack('i',  data[offset:offset+4])[0]
            offset += 4
            trace( "Labeled Marker Count:", labeledMarkerCount )
            for i in range( 0, labeledMarkerCount ):
                id = struct.unpack('i',  data[offset:offset+4])[0]
                offset += 4
                pos = Vector3.unpack( data[offset:offset+12] )
                offset += 12
                size = FloatValue.unpack( data[offset:offset+4] )
                offset += 4

                # Version 2.6 and later
                if( ( self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] >= 6 ) or self.__natNetStreamVersion[0] > 2 or major == 0 ):
                    param, = struct.unpack( 'h', data[offset:offset+2] )
                    offset += 2
                    occluded = ( param & 0x01 ) != 0
                    pointCloudSolved = ( param & 0x02 ) != 0
                    modelSolved = ( param & 0x04 ) != 0

                # Version 3.0 and later
                if( ( self.__natNetStreamVersion[0] >= 3 ) or  major == 0 ):
                    residual, = FloatValue.unpack( data[offset:offset+4] )
                    offset += 4
                    trace( "Residual:", residual )

        # Force Plate data (version 2.9 and later)
        if( ( self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] >= 9 ) or self.__natNetStreamVersion[0] > 2 ):
            forcePlateCount = struct.unpack('i',  data[offset:offset+4])[0]
            offset += 4
            trace( "Force Plate Count:", forcePlateCount )
            for i in range( 0, forcePlateCount ):
                # ID
                forcePlateID = struct.unpack('i',  data[offset:offset+4])[0]
                offset += 4
                trace( "Force Plate", i, ":", forcePlateID )

                # Channel Count
                forcePlateChannelCount = struct.unpack('i',  data[offset:offset+4])[0]
                offset += 4

                # Channel Data
                for j in range( 0, forcePlateChannelCount ):
                    trace( "\tChannel", j, ":", forcePlateID )
                    forcePlateChannelFrameCount = struct.unpack('i',  data[offset:offset+4])[0]
                    offset += 4
                    for k in range( 0, forcePlateChannelFrameCount ):
                        forcePlateChannelVal = struct.unpack('i',  data[offset:offset+4])[0]
                        offset += 4
                        trace( "\t\t", forcePlateChannelVal )

        # Device data (version 2.11 and later)
        if( ( self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] >= 11 ) or self.__natNetStreamVersion[0] > 2 ):
            deviceCount = struct.unpack('i',  data[offset:offset+4])[0]
            offset += 4
            trace( "Device Count:", deviceCount )
            for i in range( 0, deviceCount ):
                # ID
                deviceID = struct.unpack('i',  data[offset:offset+4])[0]
                offset += 4
                trace( "Device", i, ":", deviceID )

                # Channel Count
                deviceChannelCount = struct.unpack('i',  data[offset:offset+4])[0]
                offset += 4

                # Channel Data
                for j in range( 0, deviceChannelCount ):
                    trace( "\tChannel", j, ":", deviceID )
                    deviceChannelFrameCount = struct.unpack('i',  data[offset:offset+4])[0]
                    offset += 4
                    for k in range( 0, deviceChannelFrameCount ):
                        deviceChannelVal = struct.unpack('i',  data[offset:offset+4])[0]
                        offset += 4
                        trace( "\t\t", deviceChannelVal )
						       
        # Timecode            
        timecode = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4
        timecodeSub = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4

        # Timestamp (increased to double precision in 2.7 and later)
        if( ( self.__natNetStreamVersion[0] == 2 and self.__natNetStreamVersion[1] >= 7 ) or self.__natNetStreamVersion[0] > 2 ):
            timestamp, = DoubleValue.unpack( data[offset:offset+8] )
            offset += 8
        else:
            timestamp, = FloatValue.unpack( data[offset:offset+4] )
            offset += 4

        # Hires Timestamp (Version 3.0 and later)
        if( ( self.__natNetStreamVersion[0] >= 3 ) or  major == 0 ):
            stampCameraExposure = struct.unpack('ii', data[offset:offset+8])[0]
            offset += 8
            stampDataReceived = struct.unpack('ii', data[offset:offset+8])[0]
            offset += 8
            stampTransmit = struct.unpack('ii', data[offset:offset+8])[0]
            offset += 8
           
        # Frame parameters
        param, = struct.unpack( 'h', data[offset:offset+2] )
        isRecording = ( param & 0x01 ) != 0
        trackedModelsChanged = ( param & 0x02 ) != 0
        offset += 2

        # Send information to any listener.
        if self.newFrameListener is not None:
            self.newFrameListener( frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                                  labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged )

    # Unpack a marker set description packet
    def __unpackMarkerSetDescription( self, data ):
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition( b'\0' )
        offset += len( name ) + 1
        trace( "Markerset Name:", name.decode( 'utf-8' ) )
        
        markerCount = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4

        for i in range( 0, markerCount ):
            name, separator, remainder = bytes(data[offset:]).partition( b'\0' )
            offset += len( name ) + 1
            trace( "\tMarker Name:", name.decode( 'utf-8' ) )
        
        return offset

    # Unpack a rigid body description packet
    def __unpackRigidBodyDescription( self, data ):
        offset = 0

        # Version 2.0 or higher
        if( self.__natNetStreamVersion[0] >= 2 ):
            name, separator, remainder = bytes(data[offset:]).partition( b'\0' )
            offset += len( name ) + 1
            trace( "\tRigidBody Name:", name.decode( 'utf-8' ) )

        id = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4

        parentID = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4

        timestamp = Vector3.unpack( data[offset:offset+12] )
        offset += 12
        
        # Version 3.0 and higher, rigid body marker information contained in description
        if (self.__natNetStreamVersion[0] >= 3 or self.__natNetStreamVersion[0] == 0 ):
            markerCount = struct.unpack('i',  data[offset:offset+4])[0] 
            offset += 4
            trace( "\tRigidBody Marker Count:", markerCount )

            markerCountRange = range( 0, markerCount )
            for marker in markerCountRange:
                markerOffset = Vector3.unpack(data[offset:offset+12])
                offset +=12
            for marker in markerCountRange:
                activeLabel = struct.unpack('i', data[offset:offset+4])[0]
                offset += 4
            
        return offset

    # Unpack a skeleton description packet
    def __unpackSkeletonDescription( self, data ):
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition( b'\0' )
        offset += len( name ) + 1
        trace( "\tMarker Name:", name.decode( 'utf-8' ) )
        
        id = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4

        rigidBodyCount = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4

        for i in range( 0, rigidBodyCount ):
            offset += self.__unpackRigidBodyDescription( data[offset:] )

        return offset

    # Unpack a data description packet
    def __unpackDataDescriptions( self, data ):
        offset = 0
        datasetCount = struct.unpack('i',  data[offset:offset+4])[0]
        offset += 4

        for i in range( 0, datasetCount ):
            type = struct.unpack('i',  data[offset:offset+4])[0]
            offset += 4
            if( type == 0 ):
                offset += self.__unpackMarkerSetDescription( data[offset:] )
            elif( type == 1 ):
                offset += self.__unpackRigidBodyDescription( data[offset:] )
            elif( type == 2 ):
                offset += self.__unpackSkeletonDescription( data[offset:] )
            
    def __dataThreadFunction( self, socket ):
        while not rospy.is_shutdown():
            # Block for input
            data, addr = socket.recvfrom( 32768 ) # 32k byte buffer size
            if( len( data ) > 0 ):
                self.__processMessage( data )
            self.r.sleep()

    def __processMessage( self, data ):
        trace( "Begin Packet\n------------\n" )

        messageID = struct.unpack('h',  data[0:2])[0]
        trace( "\nMessage ID:", messageID )
        
        packetSize = struct.unpack('h',  data[2:4])[0]
        trace( "Packet Size:", packetSize )

        offset = 4
        if( messageID == self.NAT_FRAMEOFDATA ):
            self.__unpackMocapData( data[offset:] )
        elif( messageID == self.NAT_MODELDEF ):
            self.__unpackDataDescriptions( data[offset:] )
        elif( messageID == self.NAT_PINGRESPONSE ):
            offset += 256   # Skip the sending app's Name field
            offset += 4     # Skip the sending app's Version info
            self.__natNetStreamVersion = struct.unpack( 'BBBB', data[offset:offset+4] )
            offset += 4
        elif( messageID == self.NAT_RESPONSE ):
            if( packetSize == 4 ):
                commandResponse = struct.unpack('i',  data[offset:offset+4])[0]
                offset += 4
            else:
                message, separator, remainder = bytes(data[offset:]).partition( b'\0' )
                offset += len( message ) + 1
                trace( "Command response:", message.decode( 'utf-8' ) )
        elif( messageID == self.NAT_UNRECOGNIZED_REQUEST ):
            trace( "Received 'Unrecognized request' from server" )
        elif( messageID == self.NAT_MESSAGESTRING ):
            message, separator, remainder = bytes(data[offset:]).partition( b'\0' )
            offset += len( message ) + 1
            trace( "Received message from server:", message.decode( 'utf-8' ) )
        else:
            trace( "ERROR: Unrecognized packet type" )
            
        trace( "End Packet\n----------\n" )
            
    def sendCommand( self, command, commandStr, socket, address ):
        # Compose the message in our known message format
        if( command == self.NAT_REQUEST_MODELDEF or command == self.NAT_REQUEST_FRAMEOFDATA ):
            packetSize = 0
            commandStr = ""
        elif( command == self.NAT_REQUEST ):
            packetSize = len( commandStr ) + 1
        elif( command == self.NAT_PING ):
            commandStr = "Ping"
            packetSize = len( commandStr ) + 1

        data = struct.pack('i', command)
        data += struct.pack('i', packetSize)

        data += commandStr.encode( 'utf-8' )
        data += b'\0'

        socket.sendto( data, address )
        
    def run( self ):
        # Create the data socket
        self.dataSocket = self.__createDataSocket( self.dataPort )
        if( self.dataSocket is None ):
            print( "Could not open data channel" )
            exit

        # Create the command socket
        self.commandSocket = self.__createCommandSocket()
        if( self.commandSocket is None ):
            print( "Could not open command channel" )
            exit

        # Create a separate thread for receiving data packets
        dataThread = threading.Thread( target = self.__dataThreadFunction, args = (self.dataSocket, ))

        # # Create a separate thread for receiving command packets
        commandThread = threading.Thread( target = self.__dataThreadFunction, args = (self.commandSocket, ))

        dataThread.setDaemon(True)
        commandThread.setDaemon(True)

        dataThread.start()
        commandThread.start()

        self.sendCommand( self.NAT_REQUEST_MODELDEF, "", self.commandSocket, (self.serverIPAddress, self.commandPort) )

        try:
            while True:
                time.sleep(0.01)
        except KeyboardInterrupt as e:
            print(e)


if __name__=="__main__":
    # This will create a new NatNet client
    streamingClient = NatNetClient()

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    try:
        print("start streaming")
        streamingClient.run()
    except rospy.ROSInterruptException as re:
        print(re)
        sys.exit()
    except Exception as e:
        print(e)
        sys.exit()
    
