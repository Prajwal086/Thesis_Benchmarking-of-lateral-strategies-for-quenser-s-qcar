#!/usr/bin/env python3
"""
PC-side relay for the off-board QCar architecture.

RUN THIS FILE on the ROS 2 Humble laptop (adapter logic embedded).

QCar -> PC:
  TCP 5557 camera -> /front_camera/image_raw
  TCP 5558 telemetry -> /state/velocity and QCar telemetry topics

PC -> QCar:
  consumes embedded adapter output through ROS callbacks:
    /hardware/throttle_cmd
    /hardware/servo_cmd
    /hardware/commanded_speed
    /hardware/commanded_steering
  and sends those commands over TCP 5558.

Lane detection, controller, adapter logic, and metrics logger remain on the
laptop ROS graph. No perception result needs to be sent back to ROS 2 Dashing.
"""

import json
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64, Float64MultiArray, String


def _recv_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError('socket closed')
        data.extend(chunk)
    return bytes(data)


def _finite_vector3(value):
    """Parse PAL xyz vector or reject absent / malformed / nonfinite samples."""
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        xyz = tuple(float(v) for v in value[:3])
    except (TypeError, ValueError):
        return None
    return xyz if all(math.isfinite(v) for v in xyz) else None


def _decode_joint_pair(left, right, left_sign, right_sign, limit, tolerance):
    """Recover scalar vehicle action from signed Gazebo joint commands.

    Both joint values must agree after undoing the known Gazebo signs.
    Rejection is fail-closed: do not drive from an inconsistent wheel pair.
    """
    values = (left, right, left_sign, right_sign, limit, tolerance)
    if not all(math.isfinite(float(v)) for v in values):
        return None
    if abs(left_sign) < 1.e-6 or abs(right_sign) < 1.e-6:
        return None
    lhs, rhs = left / left_sign, right / right_sign
    if abs(lhs - rhs) > tolerance or max(abs(lhs), abs(rhs)) > limit:
        return None
    return 0.5 * (lhs + rhs)


def _integrate_pose(x, y, yaw, distance, yaw_rate, dt):
    """Midpoint integration; model-based relative odometry, NOT ground truth."""
    values = (x, y, yaw, distance, yaw_rate, dt)
    if not all(math.isfinite(float(v)) for v in values) or dt <= 0:
        raise ValueError('Invalid odometry input')
    half = yaw + 0.5 * yaw_rate * dt
    x += distance * math.cos(half)
    y += distance * math.sin(half)
    yaw = math.atan2(math.sin(yaw + yaw_rate * dt), math.cos(yaw + yaw_rate * dt))
    return x, y, yaw


def _camera_calibration_valid(width, height, fx, fy, cx, cy, distortion,
                              rectified=False):
    """Never invent physical camera intrinsics or assume zero distortion."""
    try:
        values = (float(width), float(height), float(fx), float(fy),
                  float(cx), float(cy))
        d = tuple(float(v) for v in distortion)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in values + d):
        return False
    w, h, fx, fy, cx, cy = values
    return (w >= 8 and h >= 8 and fx > 0 and fy > 0
            and 0 <= cx < w and 0 <= cy < h
            and (len(d) in (4, 5, 8, 12, 14) or (rectified and len(d) == 0)))



def _validate_recovery_path_xy(points, *, min_span_m=0.30, max_gap_m=0.16,
                               max_curvature_m_inv=2.5):
    """Independent, conservative structural verification of a ROS Path.

    Exact Path stamps must still match the detector's evidence in BOTH topic
    orders. This check cannot replace camera extrinsic/physical validation.
    """
    xy=np.asarray(points,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<8 or not np.isfinite(xy).all():
        return False, 'too few/nonfinite path samples'
    delta=np.diff(xy,axis=0)
    ds=np.linalg.norm(delta,axis=1)
    if np.any(ds<0.002) or np.any(ds>max_gap_m):
        return False,'missing/duplicated path segments'
    span=float(np.sum(ds))
    if span<min_span_m:
        return False,'path length below physical-recovery minimum'
    hd=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
    curv=np.abs(np.diff(hd)/np.maximum(.5*(ds[1:]+ds[:-1]),1.e-6))
    if len(curv) and (not np.isfinite(curv).all() or np.percentile(curv,95)>max_curvature_m_inv):
        return False,'unsafe curvature in received recovery Path'
    return True, 'validated Path geometry'


def _validate_v6_recovery_evidence(data, *, enabled, continuous_enabled,
                                   trusted_full_age, recovery_max_age,
                                   continuous_max_age, continuous_max_distance,
                                   min_span, min_consecutive):
    """Require fresh FULL ancestry + LIVE yellow proof (never cached-only)."""
    if not enabled or not isinstance(data,dict) or data.get('protocol')!='qcar_yellow_ground_v6':
        return False
    if (not data.get('valid') or not data.get('geometry_valid')
            or not data.get('fresh_yellow_observed') or data.get('nearfield_only')
            or data.get('visual_only')):
        return False
    try:
        age=float(data['age_sec']);distance=float(data['distance_since_full_m'])
        span=float(data['accepted_span_m']); curv=float(data['p95_curvature_m_inv'])
        consecutive=int(data['consecutive_frames']); stamp=int(data['path_stamp_ns'])
    except (TypeError,KeyError,ValueError,OverflowError):
        return False
    if not all(math.isfinite(v) for v in (age,distance,span,curv)) or stamp<=0:
        return False
    if (trusted_full_age is None or not math.isfinite(trusted_full_age)
            or not 0<=trusted_full_age<=continuous_max_age):
        return False
    if not 0<=age<=continuous_max_age or not 0<=distance<=continuous_max_distance:
        return False
    if span<min_span or curv>2.5 or consecutive<min_consecutive:
        return False
    # Legacy allows at most the original short window; extended is a *separate*
    # explicit permission. Both require continuously measured yellow.
    return bool(continuous_enabled or (age<=recovery_max_age and
                                       trusted_full_age<=recovery_max_age))


# Embedded hardware adapter. Logic is copied from
# offboard_hardware_adapter(2).py without changes to gains, rate limits,
# watchdogs or feedback. It now uses the relay's ROS Node instead of
# creating a separate process/node.


def _clamp(value, low, high):
    return max(low, min(high, value))


class EmbeddedOffboardHardwareAdapter:

    def __getattr__(self, name):
        # Delegate ROS Node APIs to the owning relay, keeping the adapter
        # control algorithm unchanged and eliminating a second ROS process.
        return getattr(self._node, name)

    def __init__(self, node):
        # Same adapter callbacks, parameters, speed PI and calibration as the
        # standalone adapter; all ROS resources now belong to the relay node.
        self._node = node

        self.declare_parameter('target_speed', 0.40)
        self.declare_parameter('use_speed_cmd', True)
        self.declare_parameter('speed_cmd_timeout', 0.75)
        self.declare_parameter('steering_cmd_timeout', 0.75)

        self.declare_parameter('max_target_speed', 1.00)
        self.declare_parameter('max_steer_cmd', 0.50)

        #self.declare_parameter('steering_gain', 1.39)
        self.declare_parameter('steering_gain', 1.39)
        self.declare_parameter('steering_trim', -0.087)
        self.declare_parameter('steering_rate_limit', 5.0)

        self.declare_parameter('throttle_limit', 0.30)
        self.declare_parameter('throttle_rate_limit', 1.0)
        self.declare_parameter('throttle_deadband', 0.040)
        self.declare_parameter('throttle_gain', 0.070)
        self.declare_parameter('reverse_deadband', 0.055)
        self.declare_parameter('reverse_gain', 0.070)

        # Velocity feedback is received from the existing off-board relay.
        # These PI gains are conservative starting values, NOT QCar-calibrated.
        self.declare_parameter('speed_pi_kp', 0.1)
        self.declare_parameter('speed_pi_ki', 0.1)
        self.declare_parameter('speed_pi_integral_limit', 0.1)
        self.declare_parameter('speed_pi_correction_limit', 0.1)
        self.declare_parameter('velocity_timeout_sec', 0.50)

        self.declare_parameter('output_rate_hz', 50.0)

        self.use_speed_cmd = bool(
            self.get_parameter('use_speed_cmd').value)
        self.speed_cmd_timeout = max(
            0.05,
            float(self.get_parameter('speed_cmd_timeout').value))
        self.steering_cmd_timeout = max(
            0.05,
            float(self.get_parameter('steering_cmd_timeout').value))

        self.max_target_speed = abs(
            float(self.get_parameter('max_target_speed').value))
        self.max_steer_cmd = abs(
            float(self.get_parameter('max_steer_cmd').value))

        self.steering_gain = float(
            self.get_parameter('steering_gain').value)
        self.steering_trim = float(
            self.get_parameter('steering_trim').value)
        self.steering_rate_limit = abs(
            float(self.get_parameter('steering_rate_limit').value))

        self.throttle_limit = abs(
            float(self.get_parameter('throttle_limit').value))
        self.throttle_rate_limit = abs(
            float(self.get_parameter('throttle_rate_limit').value))
        self.throttle_deadband = abs(
            float(self.get_parameter('throttle_deadband').value))
        self.throttle_gain = abs(
            float(self.get_parameter('throttle_gain').value))
        self.reverse_deadband = abs(
            float(self.get_parameter('reverse_deadband').value))
        self.reverse_gain = abs(
            float(self.get_parameter('reverse_gain').value))

        self.speed_pi_kp = max(0.0, float(self.get_parameter('speed_pi_kp').value))
        self.speed_pi_ki = max(0.0, float(self.get_parameter('speed_pi_ki').value))
        self.speed_pi_integral_limit = abs(float(
            self.get_parameter('speed_pi_integral_limit').value))
        self.speed_pi_correction_limit = abs(float(
            self.get_parameter('speed_pi_correction_limit').value))
        self.velocity_timeout_sec = max(
            0.05, float(self.get_parameter('velocity_timeout_sec').value))

        if abs(self.steering_gain) < 1.0e-9:
            raise ValueError('steering_gain must be non-zero')

        self._steering_cmd = 0.0
        self._speed_cmd = 0.0
        self._last_steering_time = None
        self._last_speed_time = None

        self._current_throttle = 0.0
        self._current_steering = 0.0
        self._last_loop_time = time.monotonic()

        self._measured_velocity = None
        self._last_velocity_time = None
        self._speed_pi_integral = 0.0
        self._pi_last_direction = 0

        self._steering_sub = self.create_subscription(
            Float64,
            '/steering_cmd',
            self._steering_cb,
            10,
        )
        self._speed_sub = self.create_subscription(
            Float64,
            '/speed_cmd',
            self._speed_cb,
            10,
        )
        self._velocity_sub = self.create_subscription(
            Float64,
            '/state/velocity',
            self._velocity_cb,
            10,
        )

        self._throttle_pub = self.create_publisher(
            Float64, '/hardware/throttle_cmd', 10
        )
        self._servo_pub = self.create_publisher(
            Float64, '/hardware/servo_cmd', 10
        )
        self._speed_pub = self.create_publisher(
            Float64, '/hardware/commanded_speed', 10
        )
        self._steering_pub = self.create_publisher(
            Float64, '/hardware/commanded_steering', 10
        )

        rate = max(
            5.0,
            float(self.get_parameter('output_rate_hz').value))
        self._timer = self.create_timer(
            1.0 / rate,
            self._step,
        )

        self.get_logger().info(
            'Laptop off-board QCar adapter started: '
            'use_speed_cmd={}, fallback target={:.2f}m/s.'.format(
                self.use_speed_cmd,
                float(self.get_parameter('target_speed').value),
            )
        )

    def _steering_cb(self, msg):
        value = float(msg.data)
        if math.isfinite(value):
            self._steering_cmd = _clamp(
                value,
                -self.max_steer_cmd,
                self.max_steer_cmd,
            )
            self._last_steering_time = time.monotonic()

    def _speed_cb(self, msg):
        value = float(msg.data)
        if math.isfinite(value):
            self._speed_cmd = _clamp(
                value,
                -self.max_target_speed,
                self.max_target_speed,
            )
            self._last_speed_time = time.monotonic()

    def _velocity_cb(self, msg):
        value = float(msg.data)
        if math.isfinite(value):
            self._measured_velocity = value
            self._last_velocity_time = time.monotonic()

    def _reset_pi(self):
        self._speed_pi_integral = 0.0
        self._pi_last_direction = 0

    @staticmethod
    def _rate_limit(current, target, rate_per_sec, dt):
        if rate_per_sec <= 0.0:
            return target
        max_step = rate_per_sec * dt
        return current + _clamp(
            target - current,
            -max_step,
            max_step,
        )

    def _steering_to_servo(self, steering):
        servo = (
            float(steering) / self.steering_gain
            + self.steering_trim
        )
        return _clamp(servo, -0.50, 0.50)

    def _speed_to_throttle(self, speed, dt):
        """Existing feedforward calibration + encoder-velocity PI correction."""
        speed = float(speed)
        if abs(speed) < 1.0e-3:
            self._reset_pi()
            return 0.0

        if speed > 0.0:
            throttle_ff = self.throttle_deadband + self.throttle_gain * speed
            direction = 1
        else:
            throttle_ff = -(
                self.reverse_deadband + self.reverse_gain * abs(speed)
            )
            direction = -1

        if direction != self._pi_last_direction:
            self._speed_pi_integral = 0.0
            self._pi_last_direction = direction

        # _step() guarantees a recent encoder-velocity measurement.
        error = speed - self._measured_velocity
        proportional = self.speed_pi_kp * error
        proposed_integral = _clamp(
            self._speed_pi_integral + self.speed_pi_ki * error * dt,
            -self.speed_pi_integral_limit,
            self.speed_pi_integral_limit,
        )
        proposed_correction = proportional + proposed_integral
        bounded_correction = _clamp(
            proposed_correction,
            -self.speed_pi_correction_limit,
            self.speed_pi_correction_limit,
        )
        proposed_throttle = throttle_ff + bounded_correction

        # Anti-windup: refuse integration that would worsen saturation.
        correction_saturated = (
            proposed_correction > self.speed_pi_correction_limit and error > 0.0
        ) or (
            proposed_correction < -self.speed_pi_correction_limit and error < 0.0
        )
        throttle_saturated = (
            proposed_throttle > self.throttle_limit and error > 0.0
        ) or (
            proposed_throttle < -self.throttle_limit and error < 0.0
        )
        if not (correction_saturated or throttle_saturated):
            self._speed_pi_integral = proposed_integral

        correction = _clamp(
            proportional + self._speed_pi_integral,
            -self.speed_pi_correction_limit,
            self.speed_pi_correction_limit,
        )
        throttle = throttle_ff + correction
        # Never actively command reverse torque to correct forward overspeed.
        if direction > 0:
            throttle = max(0.0, throttle)
        else:
            throttle = min(0.0, throttle)
        return _clamp(throttle, -self.throttle_limit, self.throttle_limit)

    def _publish_float(self, pub, value):
        msg = Float64()
        msg.data = float(value)
        pub.publish(msg)

    def _step(self):
        now = time.monotonic()
        dt = _clamp(
            now - self._last_loop_time,
            1.0e-4,
            0.25,
        )
        self._last_loop_time = now

        steering_fresh = (
            self._last_steering_time is not None
            and now - self._last_steering_time
            <= self.steering_cmd_timeout
        )

        if self.use_speed_cmd:
            speed_fresh = (
                self._last_speed_time is not None
                and now - self._last_speed_time
                <= self.speed_cmd_timeout
            )
            requested_speed = (
                self._speed_cmd if speed_fresh else 0.0
            )
        else:
            requested_speed = _clamp(
                float(self.get_parameter('target_speed').value),
                -self.max_target_speed,
                self.max_target_speed,
            )

        # Encoder feedback must be fresh before closed-loop throttle is sent.
        # Keep the existing relay topics and stop behaviour unchanged.
        velocity_fresh = (
            self._last_velocity_time is not None
            and now - self._last_velocity_time <= self.velocity_timeout_sec
        )
        if not velocity_fresh:
            requested_speed = 0.0

        # Steering is the primary controller watchdog.
        if not steering_fresh:
            requested_speed = 0.0
            requested_steering = 0.0

            # Stop immediately, not through a slow ramp.
            self._current_throttle = 0.0
            self._current_steering = 0.0
            self._reset_pi()
        else:
            requested_steering = self._steering_cmd

            # If controller explicitly requests stop, apply it immediately.
            if abs(requested_speed) < 1.0e-3:
                self._current_throttle = 0.0
                self._reset_pi()
            else:
                desired_throttle = self._speed_to_throttle(
                    requested_speed, dt)
                self._current_throttle = self._rate_limit(
                    self._current_throttle,
                    desired_throttle,
                    self.throttle_rate_limit,
                    dt,
                )

            self._current_steering = self._rate_limit(
                self._current_steering,
                requested_steering,
                self.steering_rate_limit,
                dt,
            )

        self._current_steering = _clamp(
            self._current_steering,
            -self.max_steer_cmd,
            self.max_steer_cmd,
        )

        servo = self._steering_to_servo(
            self._current_steering)

        self._publish_float(
            self._throttle_pub,
            self._current_throttle)
        self._publish_float(
            self._servo_pub,
            servo)
        self._publish_float(
            self._speed_pub,
            requested_speed)
        self._publish_float(
            self._steering_pub,
            self._current_steering)



class PCQCarOffboardRelay(Node):

    def __init__(self):
        super().__init__('pc_qcar_offboard_relay')

        self.declare_parameter('qcar_ip', '')
        self.declare_parameter('camera_port', 5557)
        self.declare_parameter('control_port', 5558)
        self.declare_parameter('command_rate_hz', 30.0)
        self.declare_parameter('command_timeout_sec', 0.50)

        # Optional Gazebo-compatible rcognita hardware interface. Opt-in:
        # initial startup must not actuate the vehicle.
        self.declare_parameter('enable_rcognita_interface', True)
        self.declare_parameter('enable_hardware_commands', True)
        self.declare_parameter('require_lane_ready', True)
        self.declare_parameter('lane_ready_timeout_sec', 0.75)
        # Experimental, fail-closed; false keeps original FULL-only behavior.
        self.declare_parameter('allow_single_boundary_recovery', False)
        self.declare_parameter('recovery_max_age_sec', 0.30)
        self.declare_parameter('recovery_path_timeout_sec', 1.0)
        self.declare_parameter('recovery_max_speed_mps', 0.20)
        # Separate, explicit physical-drive permission. Default is OFF.
        # Never rely on long-lived cached geometry alone for motion.
        self.declare_parameter('allow_continuous_single_boundary_drive', False)
        # Independent physical-calibration acknowledgment. A path's validity
        # cannot be delegated solely to the sending detector.
        self.declare_parameter('recovery_ground_geometry_verified', False)
        self.declare_parameter('expected_recovery_lane_width_m', 0.25)
        self.declare_parameter('expected_recovery_camera_height_m', 0.08)
        # PROVISIONAL: change ONLY after measuring/verifying actual pitch.
        self.declare_parameter('expected_recovery_pitch_rad', -0.08)
        self.declare_parameter('continuous_recovery_max_sec', 1.50)
        self.declare_parameter('continuous_recovery_max_distance_m', 0.45)
        self.declare_parameter('continuous_recovery_min_span_m', 0.30)
        self.declare_parameter('continuous_recovery_min_consecutive', 4)
        self.declare_parameter('recovery_max_measured_speed_mps', 0.35)

        self.declare_parameter('camera_timeout_sec', 0.75)
        self.declare_parameter('telemetry_timeout_sec', 0.50)
        self.declare_parameter('joint_cmd_timeout_sec', 0.35)
        self.declare_parameter('gazebo_wheel_radius_m', 0.034)
        self.declare_parameter('drive_left_sign', -1.0)
        self.declare_parameter('drive_right_sign', 1.0)
        self.declare_parameter('steering_left_sign', 1.0)
        self.declare_parameter('steering_right_sign', 1.0)
        self.declare_parameter('max_rcognita_speed_mps', 1.0)
        self.declare_parameter('max_rcognita_steering_rad', 0.50)
        self.declare_parameter('max_drive_pair_disagreement_radps', 0.50)
        self.declare_parameter('max_steer_pair_disagreement_rad', 0.04)

        # IMU odometry is RELATIVE. Calibration requires an explicit opt-in.
        self.declare_parameter('enable_imu_odometry', True)
        self.declare_parameter('gyro_axis_validated', True)
        self.declare_parameter('gyro_yaw_axis', 2)
        self.declare_parameter('gyro_yaw_sign', 1.0)
        self.declare_parameter('gyro_rad_per_unit', 1.0)
        self.declare_parameter('gyro_stationary_bias', 0.0044315)
        self.declare_parameter('encoder_meters_per_count', 0.01977 / 2880.0)
        self.declare_parameter('encoder_direction', 1.0)
        self.declare_parameter('odom_initial_x', 0.0)
        self.declare_parameter('odom_initial_y', 0.0)
        self.declare_parameter('odom_initial_yaw', 0.0)

        # REAL physical camera calibration (not Gazebo default intrinsics).
        self.declare_parameter('camera_fx', 0.0)
        self.declare_parameter('camera_fy', 0.0)
        self.declare_parameter('camera_cx', 0.0)
        self.declare_parameter('camera_cy', 0.0)
        self.declare_parameter('camera_distortion_csv', '')
        self.declare_parameter('rectify_camera_frames', True)
        self.declare_parameter('camera_is_rectified', True)
        self.declare_parameter('camera_distortion_model', 'plumb_bob')
        self.declare_parameter('camera_width', 820)
        self.declare_parameter('camera_height', 410)

        self.qcar_ip = str(self.get_parameter('qcar_ip').value)
        if not self.qcar_ip:
            raise RuntimeError(
                'qcar_ip is required, e.g. -p qcar_ip:=172.24.0.66'
            )

        self.camera_port = int(
            self.get_parameter('camera_port').value)
        self.control_port = int(
            self.get_parameter('control_port').value)
        self.command_timeout_sec = max(
            0.05,
            float(self.get_parameter('command_timeout_sec').value))

        p = lambda name: self.get_parameter(name).value
        if (not 0.0 < float(p('continuous_recovery_max_sec')) <= 1.50
                or not 0.0 < float(p('continuous_recovery_max_distance_m')) <= 0.45
                or not 0.30 <= float(p('continuous_recovery_min_span_m')) <= 1.0
                or int(p('continuous_recovery_min_consecutive')) < 4
                or not 0.0 < float(p('recovery_max_measured_speed_mps')) <= .40):
            raise ValueError('continuous recovery limits exceed validated guardrails')
        if (not 0.0 < float(p('recovery_max_age_sec')) <= 0.30
                or not 0.0 < float(p('recovery_path_timeout_sec')) <= 1.0
                or not 0.0 < float(p('recovery_max_speed_mps')) <= 0.30):
            raise ValueError('Recovery duration, stream timeout and speed must be positive and bounded')
        self._rcognita_enabled = bool(p('enable_rcognita_interface'))
        self._hardware_commands_enabled = bool(p('enable_hardware_commands'))
        self._imu_enabled = bool(p('enable_imu_odometry'))
        if float(p('gazebo_wheel_radius_m')) <= 0.0:
            raise ValueError('gazebo_wheel_radius_m must be positive')
        self._gyro_validated = bool(p('gyro_axis_validated'))
        if self._imu_enabled and not self._gyro_validated:
            raise RuntimeError('enable_imu_odometry requires gyro_axis_validated:=true '
                               'after physically measuring gyro yaw axis, sign, units, bias')
        self._gyro_axis = int(p('gyro_yaw_axis'))
        if self._gyro_axis not in (0, 1, 2):
            raise ValueError('gyro_yaw_axis must be 0, 1, or 2')
        self._yaw_scale = float(p('gyro_yaw_sign')) * float(p('gyro_rad_per_unit'))
        self._gyro_bias = float(p('gyro_stationary_bias'))
        self._encoder_scale = float(p('encoder_direction')) * float(p('encoder_meters_per_count'))
        if not all(math.isfinite(v) for v in (self._yaw_scale,self._gyro_bias,self._encoder_scale)):
            raise ValueError('IMU/encoder calibration parameters must be finite')
        self._pose_x = float(p('odom_initial_x'))
        self._pose_y = float(p('odom_initial_y'))
        self._pose_yaw = float(p('odom_initial_yaw'))
        self._last_encoder_odom = None
        self._last_qcar_mono = None
        self._odom_good = False
        self._last_measured_speed_mps = None
        self._last_telemetry_wall = None
        self._last_camera_wall = None
        self._last_path_wall = None
        self._last_lane_full_wall = None
        self._last_trusted_full_wall = None
        self._latest_lane_status = 0.0
        self._recovery_expected_stamp_ns = None
        self._last_recovery_evidence_wall = None
        self._recovery_evidence_kind = None
        self._recovery_evidence_reason = 'no accepted recovery evidence'
        self._last_recovery_path_wall = None
        self._last_partial_path_stamp_ns = None
        self._last_partial_path_wall = None
        self._last_raw_drive_wall = None
        self._last_raw_steer_wall = None
        self._raw_drive = None
        self._raw_steer = None
        self._last_reject_warn = 0.0
        self._published_cam_info = False
        self._camera_params = {
            'width': int(p('camera_width')), 'height': int(p('camera_height')),
            'fx': float(p('camera_fx')), 'fy': float(p('camera_fy')),
            'cx': float(p('camera_cx')), 'cy': float(p('camera_cy')),
            'distortion': [float(v) for v in str(p('camera_distortion_csv')).split(',') if v.strip()],
            'rectify': bool(p('rectify_camera_frames')),
            'rectified': bool(p('camera_is_rectified')),
            'model': str(p('camera_distortion_model')),
        }
        self._cam_calibrated = _camera_calibration_valid(
            self._camera_params['width'], self._camera_params['height'],
            self._camera_params['fx'], self._camera_params['fy'],
            self._camera_params['cx'], self._camera_params['cy'],
            self._camera_params['distortion'], self._camera_params['rectified'])
        self._cam_geometry_matched = False
        self._camera_geometry_safe = bool(self._camera_params['rectified']
                                         or self._camera_params['rectify'])
        self._last_bad_telemetry_log = 0.0

        image_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._bridge = CvBridge()

        # QCar camera -> ROS.
        self._image_pub = self.create_publisher(
            Image, '/front_camera/image_raw', image_qos
        )

        self._info_pub = self.create_publisher(
            CameraInfo, '/front_camera/camera_info', 10)
        self._imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self._imu_raw_pub = self.create_publisher(String, '/qcar/raw_imu', 10)
        self._odom_pub = self.create_publisher(Odometry, '/odom/bot', 10)
        self._odom_speed_pub = self.create_publisher(
            Float64, '/qcar/odom_forward_speed', 10)

        # QCar telemetry -> ROS.
        self._velocity_pub = self.create_publisher(
            Float64, '/state/velocity', 10
        )
        self._encoder_pub = self.create_publisher(
            Float64, '/qcar/motor_encoder', 10
        )
        self._battery_pub = self.create_publisher(
            Float64, '/qcar/battery_voltage', 10
        )
        self._current_pub = self.create_publisher(
            Float64, '/qcar/motor_current', 10
        )
        self._applied_throttle_pub = self.create_publisher(
            Float64, '/hardware/applied_throttle', 10
        )
        self._applied_servo_pub = self.create_publisher(
            Float64, '/hardware/applied_servo', 10
        )

        # Final off-board adapter output -> relay.
        self._throttle = 0.0
        self._servo = -0.087
        self._target_speed = 0.0
        self._kinematic_steering = 0.0

        self._last_throttle_time = None
        self._last_servo_time = None
        self._cmd_lock = threading.Lock()

        self._throttle_sub = self.create_subscription(
            Float64,
            '/hardware/throttle_cmd',
            self._throttle_cb,
            10,
        )
        self._servo_sub = self.create_subscription(
            Float64,
            '/hardware/servo_cmd',
            self._servo_cb,
            10,
        )
        self._speed_sub = self.create_subscription(
            Float64,
            '/hardware/commanded_speed',
            self._speed_cb,
            10,
        )
        self._steering_sub = self.create_subscription(
            Float64,
            '/hardware/commanded_steering',
            self._steering_cb,
            10,
        )

        if self._rcognita_enabled:
            self._ros_drive_sub = self.create_subscription(
                Float64MultiArray, '/drive_controller/commands',
                self._gazebo_drive_cb, 10)
            self._ros_steer_sub = self.create_subscription(
                Float64MultiArray, '/steering_controller/commands',
                self._gazebo_steer_cb, 10)
            self._lane_status_sub = self.create_subscription(
                Float64, '/lane_detection/status', self._lane_status_cb, 10)
            self._path_sub = self.create_subscription(
                Path, '/lane_detection/reference_path', self._lane_path_cb, 10)
            self._recovery_sub = self.create_subscription(
                String, '/lane_detection/recovery_evidence',
                self._recovery_evidence_cb, 10)
            self._speed_command_pub = self.create_publisher(
                Float64, '/speed_cmd', 10)
            self._steering_command_pub = self.create_publisher(
                Float64, '/steering_cmd', 10)
            self._recovery_gate_pub = self.create_publisher(
                String, '/qcar/recovery_gate_status', 10)
            self._conversion_timer = self.create_timer(
                1.0 / 30.0, self._publish_converted_commands)

        # One process / one ROS node. Keep the existing /speed_cmd,
        # /steering_cmd, /state/velocity and /hardware/* topics, so numerical
        # behavior and recording interfaces stay comparable to previous runs.
        # Do NOT also launch offboard_hardware_adapter.py.
        self._embedded_adapter = EmbeddedOffboardHardwareAdapter(self)

        self._stop = threading.Event()

        # Camera connection.
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name='qcar_camera_rx',
        )
        self._camera_thread.daemon = True
        self._camera_thread.start()

        # Control/telemetry connection.
        self._control_sock = None
        self._control_sock_lock = threading.Lock()

        self._control_thread = threading.Thread(
            target=self._control_loop,
            name='qcar_control_rx',
        )
        self._control_thread.daemon = True
        self._control_thread.start()

        command_rate = max(
            5.0,
            float(self.get_parameter('command_rate_hz').value))
        self._command_timer = self.create_timer(
            1.0 / command_rate,
            self._send_command_tick,
        )

        if self._rcognita_enabled:
            self.get_logger().warn(
                'Rcognita ROS compatibility enabled; physical actuation {}. '
                'Odometry enabled {}; physical CameraInfo valid {}. '
                'DO NOT run competing controllers.'.format(
                    self._hardware_commands_enabled, self._imu_enabled,
                    self._cam_calibrated))
        if self._hardware_commands_enabled and not (self._rcognita_enabled and
                                                    self._imu_enabled and self._cam_calibrated
                                                    and self._camera_geometry_safe):
            raise RuntimeError('Hardware commands require rcognita interface, '
                               'validated IMU odometry and physical camera calibration')

        self.get_logger().info(
            'PC off-board relay connecting to {} '
            '(camera {}, control {}).'.format(
                self.qcar_ip,
                self.camera_port,
                self.control_port,
            )
        )

    # ================================================================
    # Local adapter command callbacks
    # ================================================================

    def _throttle_cb(self, msg):
        value = float(msg.data)
        if not math.isfinite(value):
            return
        with self._cmd_lock:
            self._throttle = value
            self._last_throttle_time = time.monotonic()

    def _servo_cb(self, msg):
        value = float(msg.data)
        if not math.isfinite(value):
            return
        with self._cmd_lock:
            self._servo = value
            self._last_servo_time = time.monotonic()

    def _speed_cb(self, msg):
        value = float(msg.data)
        if math.isfinite(value):
            with self._cmd_lock:
                self._target_speed = value

    def _steering_cb(self, msg):
        value = float(msg.data)
        if math.isfinite(value):
            with self._cmd_lock:
                self._kinematic_steering = value

    # ================================================================
    # Camera connection
    # ================================================================

    def _camera_loop(self):
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection(
                    (self.qcar_ip, self.camera_port),
                    timeout=5.0,
                )
                sock.setsockopt(
                    socket.IPPROTO_TCP,
                    socket.TCP_NODELAY,
                    1,
                )
                sock.settimeout(3.0)

                self.get_logger().info(
                    'Connected to QCar camera stream.')

                while not self._stop.is_set():
                    header = _recv_exact(sock, 4)
                    jpeg_len = struct.unpack('!I', header)[0]
                    if jpeg_len <= 0 or jpeg_len > 5 * 1024 * 1024:
                        raise RuntimeError(
                            'Invalid JPEG length {}'.format(jpeg_len))

                    jpeg = _recv_exact(sock, jpeg_len)
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    if (self._cam_calibrated
                            and frame.shape[1] == self._camera_params['width']
                            and frame.shape[0] == self._camera_params['height']
                            and self._camera_params['rectify']
                            and self._camera_params['distortion']):
                        # The lane detector itself does not undistort pixels.
                        # Use calibrated K and distortion to publish rectified
                        # camera frames BEFORE computing a metric centerline.
                        c = self._camera_params
                        K = np.array([[c['fx'],0.,c['cx']],
                                      [0.,c['fy'],c['cy']],
                                      [0.,0.,1.]], dtype=np.float64)
                        frame = cv2.undistort(
                            frame, K, np.asarray(c['distortion'], dtype=np.float64))

                    msg = self._bridge.cv2_to_imgmsg(
                        frame, encoding='bgr8')
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = 'qcar_front_camera'
                    self._last_camera_wall = time.monotonic()
                    self._cam_geometry_matched = (
                        frame.shape[1] == self._camera_params['width']
                        and frame.shape[0] == self._camera_params['height'])
                    self._image_pub.publish(msg)
                    if self._cam_calibrated and self._cam_geometry_matched:
                        self._publish_camera_info(msg.header.stamp)

            except Exception as exc:
                if not self._stop.is_set():
                    self.get_logger().warn(
                        'Camera link lost: {}. Retrying...'.format(exc))
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

            if not self._stop.is_set():
                time.sleep(1.0)

    # ================================================================
    # Control + telemetry connection
    # ================================================================

    def _control_loop(self):
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection(
                    (self.qcar_ip, self.control_port),
                    timeout=5.0,
                )
                sock.setsockopt(
                    socket.IPPROTO_TCP,
                    socket.TCP_NODELAY,
                    1,
                )
                sock.settimeout(1.0)

                with self._control_sock_lock:
                    self._control_sock = sock

                self.get_logger().info(
                    'Connected to QCar actuator/telemetry link.')

                buf = b''
                while not self._stop.is_set():
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        continue

                    if not chunk:
                        raise ConnectionError('control socket closed')

                    buf += chunk
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line.decode('utf-8'))
                        except Exception:
                            continue
                        self._publish_telemetry(data)

            except Exception as exc:
                if not self._stop.is_set():
                    self.get_logger().warn(
                        'Control link lost: {}. Retrying...'.format(exc))
            finally:
                with self._control_sock_lock:
                    if self._control_sock is sock:
                        self._control_sock = None

                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

            if not self._stop.is_set():
                time.sleep(1.0)

    def _send_command_tick(self):
        now = time.monotonic()

        with self._cmd_lock:
            throttle = float(self._throttle)
            servo = float(self._servo)
            target_speed = float(self._target_speed)
            steering = float(self._kinematic_steering)
            throttle_time = self._last_throttle_time
            servo_time = self._last_servo_time

        fresh = (
            throttle_time is not None
            and servo_time is not None
            and now - throttle_time <= self.command_timeout_sec
            and now - servo_time <= self.command_timeout_sec
        )

        if not fresh:
            throttle = 0.0
            servo = -0.087
            target_speed = 0.0
            steering = 0.0

        data = {
            'throttle': throttle,
            'servo_steering': servo,
            'target_speed': target_speed,
            'kinematic_steering': steering,
            't_pc': time.time(),
        }
        payload = (
            json.dumps(data, separators=(',', ':')) + '\n'
        ).encode('utf-8')

        with self._control_sock_lock:
            sock = self._control_sock
            if sock is None:
                return
            try:
                sock.sendall(payload)
            except OSError:
                try:
                    sock.close()
                except OSError:
                    pass
                self._control_sock = None

    # ================================================================
    # Telemetry -> ROS
    # ================================================================

    @staticmethod
    def _publish_float(pub, value):
        msg = Float64()
        msg.data = float(value)
        pub.publish(msg)

    def _publish_telemetry(self, data):
        try:
            self._publish_float(
                self._velocity_pub,
                data.get('velocity', 0.0))
            self._publish_float(
                self._encoder_pub,
                data.get('encoder', 0.0))
            self._publish_float(
                self._battery_pub,
                data.get('battery', 0.0))
            self._publish_float(
                self._current_pub,
                data.get('motor_current', 0.0))
            self._publish_float(
                self._applied_throttle_pub,
                data.get('applied_throttle', 0.0))
            self._publish_float(
                self._applied_servo_pub,
                data.get('applied_servo', -0.087))
        except (TypeError, ValueError):
            return
        try:
            v=float(data.get('velocity',float('nan')))
            self._last_measured_speed_mps = v if math.isfinite(v) else None
        except (TypeError,ValueError):
            self._last_measured_speed_mps = None
        self._last_telemetry_wall = time.monotonic()
        self._publish_imu_and_odom(data)

    # ================================================================
    # Physical camera intrinsics and IMU/encoder pose estimate
    # ================================================================

    def _publish_camera_info(self, stamp):
        c = self._camera_params
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = 'qcar_front_camera'
        info.width, info.height = c['width'], c['height']
        info.distortion_model = c['model']
        # Rectified frames carry zero distortion, never the old raw coefficients.
        info.d = ([0.0]*5 if c['rectify'] or c['rectified']
                  else list(c['distortion']))
        info.k = [c['fx'],0.0,c['cx'], 0.0,c['fy'],c['cy'], 0.0,0.0,1.0]
        info.r = [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]
        info.p = [c['fx'],0.0,c['cx'],0.0,
                  0.0,c['fy'],c['cy'],0.0,
                  0.0,0.0,1.0,0.0]
        self._info_pub.publish(info)

    def _publish_imu_and_odom(self, data):
        now = time.monotonic()
        gyro = _finite_vector3(data.get('gyroscope'))
        accel = _finite_vector3(data.get('accelerometer'))
        if gyro is not None or accel is not None:
            # Raw PAL readings are deliberately NOT labelled with SI units.
            # Use this topic to verify axes and units with the motors disabled.
            raw = String()
            raw.data = json.dumps({'gyroscope': gyro,
                                   'accelerometer': accel,
                                   't_qcar': data.get('t_qcar')})
            self._imu_raw_pub.publish(raw)

        if self._imu_enabled and gyro is not None:
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'qcar_base_imu_yaw'
            # Only validated body yaw rate is reported. There is no absolute
            # orientation, and PAL accelerometer axes/units are not calibrated.
            msg.orientation_covariance[0] = -1.0
            msg.linear_acceleration_covariance[0] = -1.0
            msg.angular_velocity.z = (self._yaw_scale * gyro[self._gyro_axis]
                                       - self._gyro_bias)
            msg.angular_velocity_covariance[0] = 1.e6
            msg.angular_velocity_covariance[4] = 1.e6
            msg.angular_velocity_covariance[8] = 0.05  # provisional yaw variance
            self._imu_pub.publish(msg)

        if not self._imu_enabled or gyro is None:
            self._odom_good = False
            return
        try:
            encoder = float(data['encoder'])
            remote_time = float(data['t_qcar_monotonic'])
            if not math.isfinite(encoder) or not math.isfinite(remote_time):
                raise ValueError('invalid encoder/remote time')
            if self._last_encoder_odom is None or self._last_qcar_mono is None:
                self._last_encoder_odom = encoder
                self._last_qcar_mono = remote_time
                self._odom_good = False  # first sample only establishes origin
                return
            dt = remote_time - self._last_qcar_mono
            distance = (encoder - self._last_encoder_odom) * self._encoder_scale
            self._last_encoder_odom = encoder
            self._last_qcar_mono = remote_time
            if not (0.005 <= dt <= 0.35) or abs(distance) > 0.50:
                self._odom_good = False
                return
            yaw_rate = self._yaw_scale * gyro[self._gyro_axis] - self._gyro_bias
            self._pose_x, self._pose_y, self._pose_yaw = _integrate_pose(
                self._pose_x,self._pose_y,self._pose_yaw,distance,yaw_rate,dt)
            forward_speed = distance/dt
            stamp = self.get_clock().now().to_msg()
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x = float(self._pose_x)
            odom.pose.pose.position.y = float(self._pose_y)
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation = Quaternion(
                x=0.0,y=0.0,z=math.sin(0.5*self._pose_yaw),
                w=math.cos(0.5*self._pose_yaw))
            odom.twist.twist.linear.x = float(forward_speed)
            odom.twist.twist.linear.y = 0.0
            odom.twist.twist.angular.z = float(yaw_rate)
            # Nonzero approximate covariances signal that this is dead reckoning.
            # Increase position variance with distance travelled (heuristic).
            if not hasattr(self, '_distance_travelled'):
                self._distance_travelled = 0.0
            self._distance_travelled += abs(distance)
            pos_var = 0.02 + 0.03*self._distance_travelled
            yaw_var = 0.02 + 0.01*self._distance_travelled
            odom.pose.covariance[0] = pos_var
            odom.pose.covariance[7] = pos_var
            odom.pose.covariance[35] = yaw_var
            odom.twist.covariance[0] = 0.02
            odom.twist.covariance[35] = 0.05
            self._odom_pub.publish(odom)
            self._publish_float(self._odom_speed_pub, forward_speed)
            self._odom_good = True
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._odom_good = False
            if now - self._last_bad_telemetry_log > 2.0:
                self.get_logger().warn('No trustworthy IMU odometry: {}'.format(exc))
                self._last_bad_telemetry_log = now

    # ================================================================
    # Same Gazebo joint topic interface, mapped to existing hardware adapter
    # ================================================================

    def _gazebo_drive_cb(self, msg):
        if len(msg.data) != 2:
            self._raw_drive = None
            return
        p = lambda name: float(self.get_parameter(name).value)
        wheel_rate = _decode_joint_pair(
            float(msg.data[0]), float(msg.data[1]),
            p('drive_left_sign'), p('drive_right_sign'),
            p('max_rcognita_speed_mps')/p('gazebo_wheel_radius_m'),
            p('max_drive_pair_disagreement_radps'))
        self._raw_drive = (None if wheel_rate is None else
                           wheel_rate*p('gazebo_wheel_radius_m'))
        self._last_raw_drive_wall = time.monotonic() if self._raw_drive is not None else None

    def _gazebo_steer_cb(self, msg):
        if len(msg.data) != 2:
            self._raw_steer = None
            return
        p = lambda name: float(self.get_parameter(name).value)
        # Rcognita historically allows 0.5236 rad while the physical adapter
        # allows 0.50 rad. Decode the valid QCar/controller range first, then
        # saturate to the adapter range. Do not turn a 0.51 rad corner command
        # into a zero-command dropout. Pair consistency/finite checks remain.
        steering = _decode_joint_pair(
            float(msg.data[0]), float(msg.data[1]),
            p('steering_left_sign'), p('steering_right_sign'),
            max(p('max_rcognita_steering_rad'), 0.5236),
            p('max_steer_pair_disagreement_rad'))
        limit = abs(p('max_rcognita_steering_rad'))
        self._raw_steer = (None if steering is None else
                           max(-limit, min(limit, steering)))
        self._last_raw_steer_wall = time.monotonic() if self._raw_steer is not None else None

    def _lane_status_cb(self, msg):
        status = float(msg.data)
        now = time.monotonic()
        if not math.isfinite(status):
            status = 0.0
        previous = self._latest_lane_status
        self._latest_lane_status = status
        if status >= 3.0:
            self._last_lane_full_wall = now
            self._last_recovery_path_wall = None
            self._recovery_expected_stamp_ns = None
            self._last_partial_path_stamp_ns = None
            self._last_partial_path_wall = None
        else:
            self._last_lane_full_wall = None
            if status != 2.0 or previous != 2.0:
                self._last_recovery_path_wall = None
                self._recovery_expected_stamp_ns = None
                self._last_partial_path_stamp_ns = None
                self._last_partial_path_wall = None

    def _recovery_evidence_cb(self, msg):
        """Accept only a fresh Path plus independent, calibrated evidence.

        Evidence is not a control command. It is matched by Path timestamp,
        with independent path/fit/speed checks and explicit physical approval.
        """
        self._recovery_expected_stamp_ns = None
        self._last_recovery_path_wall = None
        self._recovery_evidence_kind = None
        self._recovery_evidence_reason = 'recovery evidence rejected'
        try:
            data = json.loads(msg.data)
            now = time.monotonic()
            p = lambda name: self.get_parameter(name).value
            if not isinstance(data, dict):
                self._recovery_evidence_reason = 'evidence is not an object'
                return
            # Physical mounting calibration must be validated separately at
            # BOTH nodes. There is no code switch that validates a real camera.
            if not (bool(p('recovery_ground_geometry_verified'))
                    and bool(data.get('verified_ground_geometry'))):
                self._recovery_evidence_reason = 'camera ground geometry not verified at detector and relay'
                return
            try:
                lane_m = float(data['measured_lane_width_m'])
                height_m = float(data['ground_camera_height_m'])
                pitch = float(data['ground_camera_pitch_rad'])
                fit_rms = float(data['fit_rmse_m'])
                fit_fraction = float(data['inlier_fraction'])
                normal_m = float(data['normal_offset_m'])
                cte = float(data['recovery_cte_m'])
                heading = float(data['recovery_heading_error_rad'])
            except (KeyError, TypeError, ValueError):
                self._recovery_evidence_reason = 'incomplete physical projection evidence'
                return
            if (not all(math.isfinite(v) for v in
                    (lane_m, height_m, pitch, fit_rms, fit_fraction, normal_m, cte, heading))
                    or abs(lane_m - float(p('expected_recovery_lane_width_m'))) > .005
                    or abs(height_m - float(p('expected_recovery_camera_height_m'))) > .005
                    or abs(pitch - float(p('expected_recovery_pitch_rad'))) > .005):
                self._recovery_evidence_reason = 'detector/relay camera geometry does not match'
                return
            if (abs(normal_m-lane_m/2)>.005 or abs(cte)>min(.095,lane_m/2-.02)
                    or abs(heading)>.35):
                self._recovery_evidence_reason = 'reconstructed centreline outside conservative corridor'
                return
            if (data.get('reconstruction') != 'curve_normal_v7'
                    or data.get('fit_model') not in ('line', 'arc')
                    or data.get('centre_off_camera')
                    or not 0<=fit_rms<=.055 or fit_fraction<.80):
                self._recovery_evidence_reason = 'curve fit not independently acceptable'
                return
            trusted_age = (None if self._last_trusted_full_wall is None
                           else now-self._last_trusted_full_wall)
            good = _validate_v6_recovery_evidence(
                data, enabled=bool(p('allow_single_boundary_recovery')),
                continuous_enabled=bool(p('allow_continuous_single_boundary_drive')),
                trusted_full_age=trusted_age,
                recovery_max_age=min(.30,float(p('recovery_max_age_sec'))),
                continuous_max_age=min(1.5,float(p('continuous_recovery_max_sec'))),
                continuous_max_distance=min(.45,float(p('continuous_recovery_max_distance_m'))),
                min_span=max(.30,float(p('continuous_recovery_min_span_m'))),
                min_consecutive=max(4,int(p('continuous_recovery_min_consecutive'))))
            if not good:
                self._recovery_evidence_reason = 'recovery timeout, travel, sequence, or proof invalid'
                return
            if self._latest_lane_status != 2.0:
                self._recovery_evidence_reason = 'lane status not PARTIAL'
                return
            stamp=int(data['path_stamp_ns'])
            now_ns=self.get_clock().now().nanoseconds
            if not 0<=now_ns-stamp<=350_000_000:
                self._recovery_evidence_reason = 'recovery Path header timestamp too old'
                return
            self._recovery_expected_stamp_ns=stamp
            self._last_recovery_evidence_wall=now
            self._recovery_evidence_kind=(
                'continuous' if bool(p('allow_continuous_single_boundary_drive')) else 'short')
            self._recovery_evidence_reason = 'geometry and evidence accepted; awaiting Path'
            if (self._last_partial_path_stamp_ns==stamp
                    and self._last_partial_path_wall is not None
                    and now-self._last_partial_path_wall<=.20):
                self._last_recovery_path_wall=now
                self._last_partial_path_stamp_ns=None
                self._last_partial_path_wall=None
                self._recovery_expected_stamp_ns=None
                self._recovery_evidence_reason='accepted matched recovery Path + evidence'
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            self._recovery_evidence_reason='evidence decode error: '+str(exc)[:100]
            return

    def _lane_path_cb(self, msg):
        now = time.monotonic()
        if len(msg.poses) < 3 or msg.header.frame_id != 'odom':
            self._last_path_wall = None
            self._last_recovery_path_wall = None
            self._last_partial_path_stamp_ns = None
            self._last_partial_path_wall = None
            return
        if self._latest_lane_status >= 3.0:
            self._last_path_wall = now
            # Trust FULL only when a real geometry Path was also published.
            self._last_trusted_full_wall = now
            return
        # Partial paths need independent geometry validation, then EXACT
        # correlation with fresh, confirmed yellow evidence from the detector.
        self._last_recovery_path_wall = None
        xy=[(pose.pose.position.x,pose.pose.position.y) for pose in msg.poses]
        ok,_reason=_validate_recovery_path_xy(
            xy,min_span_m=max(.30,float(self.get_parameter('continuous_recovery_min_span_m').value)))
        if not ok:
            self._recovery_evidence_reason = 'independent Path validation: '+_reason
            self._last_partial_path_stamp_ns=None
            self._last_partial_path_wall=None
            return
        stamp_ns = int(msg.header.stamp.sec)*1000000000 + int(msg.header.stamp.nanosec)
        if not 0<=self.get_clock().now().nanoseconds-stamp_ns<=350_000_000:
            return
        self._last_partial_path_stamp_ns = stamp_ns if self._latest_lane_status == 2.0 else None
        self._last_partial_path_wall = now if self._latest_lane_status == 2.0 else None
        if (self._latest_lane_status == 2.0
                and self._recovery_expected_stamp_ns == stamp_ns
                and self._last_recovery_evidence_wall is not None
                and now-self._last_recovery_evidence_wall <= 0.20):
            self._last_recovery_path_wall = now
            self._recovery_evidence_reason = 'accepted matched recovery Path + evidence'
        self._recovery_expected_stamp_ns = None

    def _publish_converted_commands(self):
        # Fail closed. Existing adapter additionally checks command freshness,
        # and the physical bridge retains its own independent TCP timeout.
        now = time.monotonic()
        p = lambda name: self.get_parameter(name).value
        fresh = lambda t, limit: t is not None and now-t <= float(limit)
        commands_ready = (
            self._hardware_commands_enabled
            and self._raw_drive is not None and self._raw_steer is not None
            and fresh(self._last_raw_drive_wall,p('joint_cmd_timeout_sec'))
            and fresh(self._last_raw_steer_wall,p('joint_cmd_timeout_sec')))
        sensor_ready = (
            self._odom_good and self._cam_calibrated and self._cam_geometry_matched
            and self._camera_geometry_safe
            and fresh(self._last_telemetry_wall,p('telemetry_timeout_sec'))
            and fresh(self._last_camera_wall,p('camera_timeout_sec')))
        full_ready = (
            self._latest_lane_status >= 3.0
            and fresh(self._last_path_wall, p('lane_ready_timeout_sec'))
            and fresh(self._last_lane_full_wall, p('lane_ready_timeout_sec')))
        max_recovery_age=(min(1.50,float(p('continuous_recovery_max_sec')))
            if bool(p('allow_continuous_single_boundary_drive')) else
            min(0.30,float(p('recovery_max_age_sec'))))
        recovery_ready = (
            bool(p('allow_single_boundary_recovery'))
            and self._latest_lane_status == 2.0
            and self._last_trusted_full_wall is not None
            and 0.0 <= now-self._last_trusted_full_wall <= max_recovery_age
            and fresh(self._last_recovery_path_wall,
                      min(0.20, float(p('recovery_path_timeout_sec')))))
        # Fail closed on missing or excessive measured speed during recovery.
        # Throttle-command limits alone do not establish physical speed limits.
        recovery_speed_safe=(
            self._last_measured_speed_mps is not None
            and abs(self._last_measured_speed_mps) <=
                float(p('recovery_max_measured_speed_mps')))
        if recovery_ready and not full_ready and not recovery_speed_safe:
            recovery_ready=False
        lane_ready = not bool(p('require_lane_ready')) or full_ready or recovery_ready
        # Reference-relay-style forwarding: diagnostic camera/lane/recovery
        # readiness is NOT an additional actuation gate. Retain the existing
        # ROS-to-hardware command conversion and its freshness checks.
        if commands_ready:
            speed, steering = float(self._raw_drive), float(self._raw_steer)
            if recovery_ready and not full_ready:
                speed = max(0., min(speed, float(p('recovery_max_speed_mps'))))
        else:
            speed, steering = 0.0, 0.0
        self._publish_float(self._speed_command_pub, speed)
        self._publish_float(self._steering_command_pub, steering)
        # Always publish explicit gate reasons; motor safety isn't inferred
        # from the visualization-only detector's /recovery_valid topic.
        gate = {
            'mode': ('REFERENCE_COMPAT' if commands_ready else 'STOP'),
            'lane_diagnostic_mode': ('FULL' if full_ready and lane_ready else
                                     'LIVE_YELLOW_RECOVERY' if recovery_ready else 'UNREADY'),
            'actuation_enabled': bool(self._hardware_commands_enabled),
            'commands_ready':bool(commands_ready),'sensors_ready':bool(sensor_ready),
            'full_ready':bool(full_ready),'recovery_ready':bool(recovery_ready),
            'lane_status':float(self._latest_lane_status),
            'recovery_evidence_kind':self._recovery_evidence_kind,
            'recovery_path_fresh':bool(fresh(self._last_recovery_path_wall,
                                           min(.20,float(p('recovery_path_timeout_sec'))))),
            'measured_speed_mps':self._last_measured_speed_mps,
            'recovery_speed_safe':bool(recovery_speed_safe),
            'recovery_evidence_reason':self._recovery_evidence_reason,
            'ground_geometry_verified':bool(p('recovery_ground_geometry_verified')),
            'full_age_sec':(None if self._last_trusted_full_wall is None
                            else round(now-self._last_trusted_full_wall,3)),
            'recovery_permission_age_limit_sec':float(max_recovery_age),
            'output_speed_mps':float(speed),
        }
        self._recovery_gate_pub.publish(String(data=json.dumps(gate,sort_keys=True)))
        if (self._hardware_commands_enabled and not commands_ready
                and now-self._last_reject_warn > 3.0):
            self.get_logger().warn('Hardware actuation blocked: commands unavailable/stale '
                                   '(diagnostics: sensors={} lane={})'.format(
                                       sensor_ready, lane_ready))
            self._last_reject_warn = now

    def destroy_node(self):
        self._stop.set()

        with self._control_sock_lock:
            sock = self._control_sock
            if sock is not None:
                # Best-effort STOP before closing.
                try:
                    stop = {
                        'throttle': 0.0,
                        'servo_steering': -0.087,
                        'target_speed': 0.0,
                        'kinematic_steering': 0.0,
                    }
                    sock.sendall(
                        (json.dumps(stop) + '\n').encode('utf-8'))
                except OSError:
                    pass
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                self._control_sock = None

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PCQCarOffboardRelay()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
