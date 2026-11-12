#!/usr/bin/env python3
"""
odom_to_tf.py

Subscribes to /odom/bot (nav_msgs/Odometry from libgazebo_ros_p3d) and:
  1. Broadcasts the single missing TF:  odom → base
  2. Re-publishes the odometry on /odom  (with correct frame_id / child_frame_id)
     so RViz Odometry display and any navigation stack can consume it.

This node does NOT conflict with robot_state_publisher, which only publishes
transforms between links WITHIN the URDF (base→hubfl, base→wheelrl, etc.).
This node exclusively owns the odom→base edge at the top of the TF tree.

Launch file entry:
    Node(
        package='qcar_gazebo',
        executable='odom_to_tf',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

CMakeLists.txt install entry:
    install(PROGRAMS scripts/odom_to_tf.py
            DESTINATION lib/${PROJECT_NAME}
            RENAME odom_to_tf)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros


class OdomToTF(Node):
    def __init__(self):
        super().__init__('odom_to_tf')

        # TF broadcaster — publishes odom → base on /tf
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        # Re-publish odometry with correct frame ids on /odom
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Subscribe to the raw odometry from libgazebo_ros_p3d
        self.sub = self.create_subscription(
            Odometry,
            '/odom/bot',
            self.odom_callback,
            10
        )

        self.get_logger().info(
            'odom_to_tf started\n'
            '  Listening : /odom/bot\n'
            '  TF        : odom → base\n'
            '  Publishing: /odom'
        )

    def odom_callback(self, msg: Odometry):
        now = msg.header.stamp  # use Gazebo sim stamp, not wall clock

        # ------------------------------------------------------------------
        # 1. Broadcast  odom → base  TF
        # ------------------------------------------------------------------
        t = TransformStamped()
        t.header.stamp    = now
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base'

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation      = msg.pose.pose.orientation

        self.broadcaster.sendTransform(t)

        # ------------------------------------------------------------------
        # 2. Re-publish odometry on /odom with correct frame metadata
        #    (p3d sometimes leaves frame_id blank or set to 'world')
        # ------------------------------------------------------------------
        odom_out = Odometry()
        odom_out.header.stamp    = now
        odom_out.header.frame_id = 'odom'   # fixed/world frame
        odom_out.child_frame_id  = 'base'   # moving frame

        odom_out.pose  = msg.pose
        odom_out.twist = msg.twist

        self.odom_pub.publish(odom_out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
