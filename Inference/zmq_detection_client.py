#! /usr/bin/env python
import rospy
from geometry_msgs.msg import Vector3
import zmq
import json

if __name__ == '__main__':
    try:
        rospy.init_node('zmq_detection_client', anonymous=False)
        pub = rospy.Publisher('/robot_detection/zmq_client', Vector3, queue_size=0)

        port = "5556"
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.connect("tcp://localhost:%s" % port)

        while not rospy.is_shutdown():
            msg = socket.recv_string()
            indicator, x, y, z = json.loads(msg)
            if indicator:
                pub.publish(Vector3(x, y, z))
    except rospy.ROSInterruptException:
        pass