import math

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class ReferencePathPublisher(Node):
    def __init__(self):
        super().__init__('reference_path_publisher')

        self.declare_parameter('path_type', 'circle')   # circle or straight
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('publish_rate', 1.0)

        # Circle parameters
        self.declare_parameter('circle_radius', 5.0)
        self.declare_parameter('circle_points', 200)
        self.declare_parameter('circle_center_x', 0.0)
        self.declare_parameter('circle_center_y', 5.0)

        # Straight line parameters
        self.declare_parameter('straight_length', 15.0)
        self.declare_parameter('straight_points', 100)
        self.declare_parameter('straight_start_x', 0.0)
        self.declare_parameter('straight_start_y', 0.0)
        self.declare_parameter('straight_heading', 0.0)  # radians

        # Mild curve parameters
        self.declare_parameter('mild_curve_length', 15.0)
        self.declare_parameter('mild_curve_amplitude', 1.5)
        self.declare_parameter('mild_curve_points', 200)
        self.declare_parameter('mild_curve_start_x', 0.0)
        self.declare_parameter('mild_curve_start_y', 0.0)

        # Infinity / figure-eight S-curve parameters
        self.declare_parameter('figure_eight_scale', 4.0)
        self.declare_parameter('figure_eight_points', 400)
        self.declare_parameter('figure_eight_center_x', 0.0)
        self.declare_parameter('figure_eight_center_y', 0.0)

        # Open S-curve matching rcognita/qcar_paths.py::make_path('s_curve').
        # The old 's_curve' ROS option remains the existing closed infinity path.
        self.declare_parameter('standalone_s_curve_radius', 2.5)
        self.declare_parameter('standalone_s_curve_points', 240)
        self.declare_parameter('standalone_s_curve_start_x', 0.0)
        self.declare_parameter('standalone_s_curve_start_y', 0.0)

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.path_pub = self.create_publisher(Path, '/reference_path', qos)

        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        timer_period = 1.0 / publish_rate if publish_rate > 0.0 else 1.0
        self.timer = self.create_timer(timer_period, self.publish_path)

        self.get_logger().info('Reference path publisher started.')
        self.get_logger().info(
            f"path_type={self.get_parameter('path_type').get_parameter_value().string_value}"
        )

    def build_circle_path(self) -> Path:
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        radius = self.get_parameter('circle_radius').get_parameter_value().double_value
        num_points = self.get_parameter('circle_points').get_parameter_value().integer_value
        center_x = self.get_parameter('circle_center_x').get_parameter_value().double_value
        center_y = self.get_parameter('circle_center_y').get_parameter_value().double_value

        path = Path()
        path.header.frame_id = frame_id
        now = self.get_clock().now().to_msg()
        path.header.stamp = now

        for i in range(num_points):
            theta = 2.0 * math.pi * i / num_points

            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)

            # Tangent heading for CCW motion
            yaw = theta + math.pi / 2.0

            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = now
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation = yaw_to_quaternion(yaw)

            path.poses.append(pose)

        return path

    def build_straight_path(self) -> Path:
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        length = self.get_parameter('straight_length').get_parameter_value().double_value
        num_points = self.get_parameter('straight_points').get_parameter_value().integer_value
        start_x = self.get_parameter('straight_start_x').get_parameter_value().double_value
        start_y = self.get_parameter('straight_start_y').get_parameter_value().double_value
        heading = self.get_parameter('straight_heading').get_parameter_value().double_value

        path = Path()
        path.header.frame_id = frame_id
        now = self.get_clock().now().to_msg()
        path.header.stamp = now

        for i in range(num_points):
            s = length * i / max(1, num_points - 1)

            x = start_x + s * math.cos(heading)
            y = start_y + s * math.sin(heading)

            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = now
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation = yaw_to_quaternion(heading)

            path.poses.append(pose)

        return path
    
    def build_mild_curve_path(self) -> Path:
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        length = self.get_parameter('mild_curve_length').get_parameter_value().double_value
        amplitude = self.get_parameter('mild_curve_amplitude').get_parameter_value().double_value
        num_points = self.get_parameter('mild_curve_points').get_parameter_value().integer_value
        start_x = self.get_parameter('mild_curve_start_x').get_parameter_value().double_value
        start_y = self.get_parameter('mild_curve_start_y').get_parameter_value().double_value

        path = Path()
        path.header.frame_id = frame_id
        now = self.get_clock().now().to_msg()
        path.header.stamp = now

        for i in range(num_points):
            t = i / max(1, num_points - 1)

            x = start_x + length * t
            y = start_y + amplitude * math.sin(math.pi * t)

            dx_dt = length
            dy_dt = amplitude * math.pi * math.cos(math.pi * t)
            yaw = math.atan2(dy_dt, dx_dt)

            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = now
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation = yaw_to_quaternion(yaw)

            path.poses.append(pose)

        return path
    
    def build_figure_eight_path(self) -> Path:
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        scale = self.get_parameter('figure_eight_scale').get_parameter_value().double_value
        num_points = self.get_parameter('figure_eight_points').get_parameter_value().integer_value
        center_x = self.get_parameter('figure_eight_center_x').get_parameter_value().double_value
        center_y = self.get_parameter('figure_eight_center_y').get_parameter_value().double_value

        path = Path()
        path.header.frame_id = frame_id
        now = self.get_clock().now().to_msg()
        path.header.stamp = now

        for i in range(num_points):
            t = 2.0 * math.pi * i / num_points

            # Lemniscate / infinity-like path
            x = center_x + scale * math.sin(t)
            y = center_y + scale * math.sin(t) * math.cos(t)

            dx_dt = scale * math.cos(t)
            dy_dt = scale * (math.cos(t) ** 2 - math.sin(t) ** 2)

            yaw = math.atan2(dy_dt, dx_dt)

            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = now
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation = yaw_to_quaternion(yaw)

            path.poses.append(pose)

        return path

    def build_s_curve_path(self) -> Path:
        """Publish the *open* S-curve generated by rcognita.qcar_paths.make_path.

        The reference math deliberately matches make_path('s_curve', points,
        radius) exactly; no import from rcognita is required by this ROS node.
        With defaults: x in [0, 15] m, y = 1.5 sin(pi*x/7.5), 240 points.
        """
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        radius = self.get_parameter('standalone_s_curve_radius').get_parameter_value().double_value
        num_points = self.get_parameter('standalone_s_curve_points').get_parameter_value().integer_value
        start_x = self.get_parameter('standalone_s_curve_start_x').get_parameter_value().double_value
        start_y = self.get_parameter('standalone_s_curve_start_y').get_parameter_value().double_value

        if num_points < 20 or radius <= 0.0:
            raise ValueError('standalone_s_curve_points >= 20 and radius > 0 are required')

        # Identical equations and sampling to rcognita/qcar_paths.py.
        x = np.linspace(0.0, 6.0 * radius, num_points)
        amplitude = 0.6 * radius
        y = amplitude * np.sin(np.pi * x / (3.0 * radius))
        heading = np.arctan2(np.gradient(y), np.gradient(x))

        path = Path()
        now = self.get_clock().now().to_msg()
        path.header.frame_id = frame_id
        path.header.stamp = now

        for x_i, y_i, yaw in zip(x, y, heading):
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = now
            pose.pose.position.x = float(x_i + start_x)
            pose.pose.position.y = float(y_i + start_y)
            pose.pose.position.z = 0.0
            pose.pose.orientation = yaw_to_quaternion(float(yaw))
            path.poses.append(pose)

        return path

    def publish_path(self):
        path_type = self.get_parameter('path_type').get_parameter_value().string_value.lower()

        if path_type == 'circle':
            path = self.build_circle_path()
        elif path_type == 'straight':
            path = self.build_straight_path()
        elif path_type == 'mild_curve':
            path = self.build_mild_curve_path()
        elif path_type in ['figure_eight', 'infinity']:
            path = self.build_figure_eight_path()
        elif path_type in ['s_curve', 'rcognita_s_curve', 'open_s_curve']:
            path = self.build_s_curve_path()
        else:
            self.get_logger().warn(
                f"Unknown path_type='{path_type}', falling back to circle."
            )
            path = self.build_circle_path()

        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)

    node = ReferencePathPublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
