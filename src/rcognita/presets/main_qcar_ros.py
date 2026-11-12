#!/usr/bin/env python3
"""Thin ROS2/Gazebo adapter for the EXACT rcognita QCar classes in main_qcar.py.

One ROS timer is the sole caller of controller.compute_action(). No controller
logic, path-error computation, simulator stepping, or animator control loop.
NOTE: The default wheel command signs mirror the uploaded legacy Gazebo preset.
Verify topics, wheel signs and odometry pose origin on your specific QCar URDF.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import math
import numpy as np
from rcognita import loggers
try:
    from presets.main_qcar import add_arguments, build
except ImportError:
    from main_qcar import add_arguments, build


def lane_path_quality(points):
    """Geometric gate/diagnostics, not a controller's CTE calculation.

    Uses fixed-frame path points and is intentionally independent of ROS, so
    malformed perception paths can be tested without a Gazebo installation.
    """
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 5 or not np.isfinite(xy).all():
        return dict(valid=False, reason='invalid/nonfinite path', points=len(xy), length_m=0.)
    segments = np.diff(xy, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths < 1.e-3):
        return dict(valid=False, reason='duplicate or millimetre-scale segments', points=len(xy), length_m=float(lengths.sum()))
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    curvatures = np.abs(np.diff(headings) / np.maximum(0.5*(lengths[:-1]+lengths[1:]), 1.e-8))
    return dict(valid=True, reason='', points=len(xy), length_m=float(lengths.sum()),
                min_segment_m=float(lengths.min()),
                p95_curvature_m_inv=float(np.percentile(curvatures, 95)) if len(curvatures) else 0.)


def lane_remaining_preview_m(points, position_xy):
    """Available arc length from nearest projection to the LAST observed point.

    This geometric safety check does not calculate CTE or change steering.
    Distance from the rear axle to the first camera point is NOT extrapolated.
    """
    xy = np.asarray(points, dtype=float)
    pos = np.asarray(position_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or pos.shape != (2,):
        return 0.0
    if not np.isfinite(xy).all() or not np.isfinite(pos).all():
        return 0.0
    delta = np.diff(xy, axis=0)
    length2 = np.einsum('ij,ij->i', delta, delta)
    if np.any(length2 <= 1e-10):
        return 0.0
    fraction = np.clip(np.einsum('ij,ij->i', pos-xy[:-1], delta)/length2, 0., 1.)
    projected = xy[:-1]+fraction[:, None]*delta
    idx = int(np.argmin(np.sum((projected-pos)**2, axis=1)))
    tangent = delta[-1] / np.sqrt(length2[-1])
    if idx == len(delta)-1 and np.dot(pos-xy[-1], tangent) > 0.:
        return 0.0
    lengths = np.sqrt(length2)
    return float(np.sum(lengths[idx+1:]) + (1.-fraction[idx])*lengths[idx])


def forward_speed_from_twist(vx, vy, yaw, twist_frame='body'):
    """Longitudinal speed in the vehicle frame from the odometry twist.

    nav_msgs/Odometry defines twist in child_frame_id ('body'), but the
    Gazebo /odom/bot plugin used in this QCar project has been verified from
    recorded odometry to emit world-frame vx/vy. World-mode projects that
    vector onto the robot's body-forward axis. This DOES NOT integrate or
    estimate vehicle position; Gazebo continues to provide the pose.

    magnitude mode computes hypot(vx, vy) and is invariant to planar frame
    rotation, but has NO direction sign: reverse and lateral slip are counted
    as positive speed. Prefer it only for forward-driving experiments.

    An explicit option avoids silently changing behavior for other odometry
    publishers which correctly use child-frame velocities.
    """
    vx, vy, yaw = float(vx), float(vy), float(yaw)
    if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(yaw)):
        raise ValueError('Nonfinite odometry velocity or heading')
    if twist_frame == 'body':
        return vx
    if twist_frame == 'world':
        return vx * math.cos(yaw) + vy * math.sin(yaw)
    if twist_frame == 'magnitude':
        return math.hypot(vx, vy)
    raise ValueError('odom_twist_frame must be body, world, or magnitude')


def main():
    """Run the ROS transport adapter with shared controller and ROS tuning options.

    The shared controller settings are documented by ``add_arguments`` in
    ``main_qcar.py``; ROS-specific units, thresholds, and coordinate offsets are
    described alongside their command-line options below.
    """
    # Import ROS only inside this ROS preset. rcognita and main_qcar.py stay ROS-free.
    import rclpy
    import csv
    from rclpy.node import Node
    from rclpy.utilities import remove_ros_args
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from nav_msgs.msg import Odometry, Path as ROSPath
    from std_msgs.msg import Float64, Float64MultiArray

    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--odom_topic',default='/odom/bot',
                        help='ROS topic supplying nav_msgs/Odometry measurements')
    parser.add_argument('--odom_twist_frame', choices=['body', 'world', 'magnitude'], default='world',
                        help='Frame of odometry twist.linear x/y: body is ROS standard; '
                             'world projects vx/vy along vehicle heading; '
                             'magnitude uses sqrt(vx**2 + vy**2), intended for forward-only motion.')
    parser.add_argument('--velocity_topic', default='/state/velocity',
                        help='Publish the same measured scalar speed sent to rcognita as std_msgs/Float64.')
    parser.add_argument('--input_mode',choices=['path','lane'],default='path',
                        help='Use a generated reference-path topic or a camera-derived lane path')
    parser.add_argument('--path_topic',default='/reference_path',
                        help='ROS topic for the external reference path in path mode')
    parser.add_argument('--lane_path_topic',default='/lane_detection/reference_path',
                        help='Camera-derived nav_msgs/Path in the odometry frame. Must be published by perception.')
    parser.add_argument('--path_timeout',type=float,default=0.5,
                        help='Maximum age in seconds of an accepted lane path before commands stop')
    parser.add_argument('--lane_min_path_length_m',type=float,default=0.10,
                        help='Reject camera paths with less total observed arc length, in metres')
    parser.add_argument('--lane_max_path_p95_curvature',type=float,default=10.0,
                        help='Reject camera paths whose 95th-percentile absolute curvature exceeds this 1/m threshold')
    parser.add_argument('--lane_path_diagnostics_sec',type=float,default=0.5,
                        help='Minimum interval in seconds between accepted/rejected path log messages')
    parser.add_argument('--lane_mpc_finite_path', action=argparse.BooleanOptionalAction,
                        default=True, help='Limit MPC to observed camera-path points and account for its endpoint')
    parser.add_argument('--lane_mpc_end_buffer_m', type=float, default=0.15,
                        help='MPC stopping buffer before the last observed lane point, in metres')
    parser.add_argument('--lane_mpc_brake_decel_mps2', type=float, default=0.40,
                        help='Deceleration in m/s^2 used to estimate safe speed near the observed path endpoint')
    parser.add_argument('--lane_mpc_overrun_weight', type=float, default=60.0,
                        help='MPC cost weight for predicted travel beyond the observed lane-path endpoint')
    parser.add_argument('--lane_mpc_min_remaining_preview_m', type=float, default=0.16,
                        help='Stop when less observed path remains ahead of the rear axle, in metres')
    parser.add_argument('--lane_diagnostics_csv', default='',
                        help='Optional sidecar CSV for path acceptance/rejection and remaining preview; '
                             'defaults to <controller_csv>.lane_quality.csv in lane mode')
    parser.add_argument('--drive_topic',default='/drive_controller/commands',
                        help='ROS topic receiving the left/right wheel angular-speed commands')
    parser.add_argument('--steering_topic',default='/steering_controller/commands',
                        help='ROS topic receiving the left/right steering-angle commands')
    parser.add_argument('--wheel_radius',type=float,default=0.034,
                        help='Drive-wheel radius in metres, used to convert linear to angular speed')
    parser.add_argument('--odom_to_rear_axle',type=float,default=-0.12765,
                        help='Signed distance from odom reference to rear axle along heading. '
                             'If odom is at geometric center, check whether -L/2 applies.')
    parser.add_argument('--drive_left_sign',type=float,default=-1.,
                        help='Multiplier applied to the left wheel-speed command to match actuator convention')
    parser.add_argument('--drive_right_sign',type=float,default=1.,
                        help='Multiplier applied to the right wheel-speed command to match actuator convention')
    parser.add_argument('--steering_left_sign',type=float,default=1.,
                        help='Multiplier applied to the left steering command to match actuator convention')
    parser.add_argument('--steering_right_sign',type=float,default=1.,
                        help='Multiplier applied to the right steering command to match actuator convention')
    parser.add_argument('--odom_timeout',type=float,default=0.5,
                        help='Stop publishing nonzero commands after this many seconds with no fresh odometry')
    parser.add_argument('--csv',default='qcar_gazebo_results.csv',
                        help='Output CSV file for controller and tracking results')
    parser.add_argument('--animate', action='store_true',
                        help='Live six-panel read-only Matplotlib GUI (does not start a second control loop)')
    parser.add_argument('--gui_refresh_hz', type=float, default=5.0,
                        help='Live GUI redraw rate in frames per second (allowed range: 0.5 to 30)')
    parser.add_argument('--car_display_mode', choices=['physical', 'icon'], default='physical',
                        help='Render the car as a physical model or a simple icon in the GUI')
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    if args.wheel_radius<=0 or args.odom_timeout<=0:
        parser.error('wheel_radius and odom_timeout must be positive')
    if args.path_timeout <= 0:
        parser.error('--path_timeout must be positive')
    if (args.lane_min_path_length_m <= 0 or args.lane_max_path_p95_curvature <= 0
            or args.lane_path_diagnostics_sec <= 0):
        parser.error('lane path validation parameters must be positive')
    if (args.lane_mpc_end_buffer_m < 0 or args.lane_mpc_brake_decel_mps2 <= 0 or
            args.lane_mpc_overrun_weight < 0 or args.lane_mpc_min_remaining_preview_m <= 0):
        parser.error('Invalid finite camera-path constraints')
    if not 0.5 <= args.gui_refresh_hz <= 30.0:
        parser.error('--gui_refresh_hz must be between 0.5 and 30')
    if args.animate:
        # Refuse an unavailable GUI before creating any ROS node that can move
        # Gazebo. Otherwise an Agg backend would briefly start then abort.
        import matplotlib
        backend = str(matplotlib.get_backend()).lower()
        if backend in ('agg', 'pdf', 'svg', 'ps', 'cairo', 'template') or 'inline' in backend:
            parser.error(f'--animate needs a Matplotlib GUI backend; found {backend!r}. '
                         'Try MPLBACKEND=TkAgg and install python3-tk.')

    my_sys,my_ctrl,path,active = build(args)
    if args.input_mode == 'lane' and args.ctrl_mode == 'MPC':
        my_ctrl.set_dynamic_path_mode(True)
        my_ctrl.set_finite_path_mode(
            args.lane_mpc_finite_path,
            end_buffer_m=args.lane_mpc_end_buffer_m,
            brake_decel_mps2=args.lane_mpc_brake_decel_mps2,
            overrun_weight=args.lane_mpc_overrun_weight,
        )
    live_animator = None
    if args.animate:
        # GUI runs on the process's MAIN thread. The single ROS executor uses
        # a background thread; only its timer calls compute_action().
        from rcognita.visuals import AnimatorQCar
        live_animator = AnimatorQCar(
            path=np.zeros((3, 2)),  # placeholder, NEVER shown as reference
            controller_name=args.ctrl_mode,
            path_name='Gazebo ' + args.input_mode,
            target_speed=args.target_speed,
            wheelbase=args.wheelbase,
            car_display_mode=args.car_display_mode,
        )
    logger=loggers.LoggerQCar(args.csv,dict(
        system='SysQCar',controller=type(my_ctrl).__name__,
        environment='ROS2_Gazebo',path_type=args.path_type,
        path_radius=args.path_radius,parameters=active,
        input_mode=args.input_mode, wheel_radius=args.wheel_radius,
        odom_twist_frame=args.odom_twist_frame,
        velocity_topic=args.velocity_topic,
        lane_mpc_finite_path=bool(args.input_mode == 'lane' and args.ctrl_mode == 'MPC' and args.lane_mpc_finite_path),
        lane_mpc_end_buffer_m=args.lane_mpc_end_buffer_m,
        lane_mpc_brake_decel_mps2=args.lane_mpc_brake_decel_mps2,
        lane_mpc_overrun_weight=args.lane_mpc_overrun_weight,
        lane_mpc_min_remaining_preview_m=args.lane_mpc_min_remaining_preview_m,
        odom_to_rear_axle=args.odom_to_rear_axle,
        drive_left_sign=args.drive_left_sign,
        drive_right_sign=args.drive_right_sign,
        steering_left_sign=args.steering_left_sign,
        steering_right_sign=args.steering_right_sign))

    class ROSPreset(Node):
        """ROS transport only. State/errors remain exclusively in my_ctrl."""
        def __init__(self):
            super().__init__('rcognita_qcar_controller')
            self.observation = None
            self.last_odom_stamp = None
            self.odom_frame = None
            self.last_path_stamp = None
            self.last_path_frame = None
            self.start_control_stamp = None
            self.last_control_stamp = None
            self.last_warning_sec = float('-inf')
            self.last_quality_log = float('-inf')
            self.last_path_quality = None
            self.accepted_path_xy = None
            self._preview_stop_logged = float('-inf')
            self._diag_file = None
            self._diag_writer = None
            if args.input_mode == 'lane':
                filename = args.lane_diagnostics_csv or args.csv + '.lane_quality.csv'
                from pathlib import Path as _Path
                _Path(filename).parent.mkdir(parents=True, exist_ok=True)
                self._diag_file = open(filename, 'w', newline='', buffering=1)
                self._diag_writer = csv.writer(self._diag_file)
                self._diag_writer.writerow(['wall_time_s','accepted','reason','path_points',
                    'path_length_m','p95_curvature_m_inv','remaining_preview_m'])
            self.velocity_pub=self.create_publisher(Float64,args.velocity_topic,10)
            self.drive_pub=self.create_publisher(Float64MultiArray,args.drive_topic,10)
            self.steer_pub=self.create_publisher(Float64MultiArray,args.steering_topic,10)
            self.sub=self.create_subscription(Odometry,args.odom_topic,self.on_odom,10)
            # Reference publishers often latch the path. Camera perception is
            # live/volatile; requesting TRANSIENT_LOCAL from a volatile publisher
            # is incompatible on some DDS implementations.
            durability = (DurabilityPolicy.TRANSIENT_LOCAL if args.input_mode == 'path'
                          else DurabilityPolicy.VOLATILE)
            path_qos=QoSProfile(depth=1,durability=durability,
                                reliability=ReliabilityPolicy.RELIABLE)
            path_topic = (args.path_topic if args.input_mode == 'path'
                          else args.lane_path_topic)
            self.path_sub=self.create_subscription(ROSPath,path_topic,self.on_path,path_qos)
            self.timer=self.create_timer(args.dt,self.control_cycle)
            self.get_logger().info(f'{args.ctrl_mode} active; [v,delta] from rcognita; '
                f'input_mode={args.input_mode}; path={path_topic}; odom={args.odom_topic}; '
                f'odom_twist_frame={args.odom_twist_frame}; one control timer')

        def on_odom(self,msg):
            self.odom_frame = msg.header.frame_id
            p=msg.pose.pose.position
            q=msg.pose.pose.orientation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            # Normalization ONLY: odom location -> rear-axle pose. NOT error computation.
            rear_x=p.x+args.odom_to_rear_axle*math.cos(yaw)
            rear_y=p.y+args.odom_to_rear_axle*math.sin(yaw)
            # ROS standard: twist in child_frame_id. However, actual Gazebo
            # /odom/bot contains WORLD vx/vy (confirmed by rosbag pose slopes).
            # The explicit CLI option selects the conversion. Magnitude is
            # direction-free: use it for forward-only driving experiments.
            try:
                longitudinal_speed = forward_speed_from_twist(
                    msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                    yaw, args.odom_twist_frame)
                if not np.isfinite([rear_x, rear_y, yaw]).all():
                    raise ValueError('Nonfinite odometry pose')
            except ValueError as exc:
                # Do not refresh the odometry watchdog on invalid measurements.
                self.get_logger().warning(f'Ignoring invalid odometry: {exc}')
                return
            self.observation=np.array([rear_x,rear_y,yaw,longitudinal_speed])
            velocity_msg = Float64()
            velocity_msg.data = float(longitudinal_speed)
            self.velocity_pub.publish(velocity_msg)
            self.last_odom_stamp=self.get_clock().now().nanoseconds*1e-9

        def on_path(self,msg):
            if len(msg.poses) < 3:
                self.get_logger().warning('Ignoring reference path with <3 poses')
                if self._diag_writer is not None:
                    self._diag_writer.writerow([self.get_clock().now().nanoseconds*1e-9,
                        0, 'fewer than three waypoints', len(msg.poses), 0., '', 0.])
                return
            xy=np.asarray([[p.pose.position.x,p.pose.position.y]
                           for p in msg.poses],dtype=float)
            # Path geometry lives in CtrlPathTracking.update_path().
            # The perception node must publish positions in the odometry frame.
            if self.odom_frame and msg.header.frame_id != self.odom_frame:
                self.get_logger().warning(
                    f'Ignoring path in frame {msg.header.frame_id!r}; expected {self.odom_frame!r}. '
                    'Transform it in the perception/reference publisher.')
                if self._diag_writer is not None:
                    self._diag_writer.writerow([self.get_clock().now().nanoseconds*1e-9,
                        0, 'path frame mismatch', len(msg.poses), '', '', 0.])
                return
            if self.last_path_frame is None:
                self.last_path_frame = msg.header.frame_id
            elif self.last_path_frame != msg.header.frame_id:
                self.get_logger().warning('Reference path frame changed; verify TF')
                if self._diag_writer is not None:
                    self._diag_writer.writerow([self.get_clock().now().nanoseconds*1e-9,
                        0, 'path frame changed', len(msg.poses), '', '', 0.])
                return
            if args.input_mode == 'lane':
                quality = lane_path_quality(xy)
                now = self.get_clock().now().nanoseconds*1e-9
                self.last_path_quality = quality
                remaining = lane_remaining_preview_m(xy, self.observation[:2]) if self.observation is not None else 0.0
                quality['remaining_preview_m'] = remaining
                rejection = (not quality.get('valid', False) or
                    quality.get('length_m', 0.) < args.lane_min_path_length_m or
                    quality.get('p95_curvature_m_inv', float('inf')) > args.lane_max_path_p95_curvature)
                if (not rejection and args.ctrl_mode == 'MPC' and args.lane_mpc_finite_path and
                        self.observation is not None and
                        remaining < args.lane_mpc_min_remaining_preview_m):
                    rejection = True
                    quality['reason'] = 'insufficient observed remaining preview'
                if self._diag_writer is not None:
                    self._diag_writer.writerow([now, int(not rejection),
                        quality.get('reason', '') or ('geometry rejected' if rejection else 'accepted'),
                        quality.get('points', 0), quality.get('length_m', 0.),
                        quality.get('p95_curvature_m_inv', float('nan')), remaining])
                if rejection:
                    if now - self.last_quality_log >= args.lane_path_diagnostics_sec:
                        self.get_logger().warning('Rejecting camera path: {}'.format(quality))
                        self.last_quality_log = now
                    # Deliberately do NOT refresh the last accepted timestamp.
                    # The existing stale-path watchdog will stop the vehicle.
                    return
                if now - self.last_quality_log >= args.lane_path_diagnostics_sec:
                    self.get_logger().info('Accepted camera path: {}'.format(quality))
                    self.last_quality_log = now
            try:
                my_ctrl.update_path(xy)
                if args.input_mode == 'lane':
                    self.accepted_path_xy = xy.copy()
                if live_animator is not None:
                    live_animator.update_live_path(xy)
                self.last_path_stamp = self.get_clock().now().nanoseconds*1e-9
            except ValueError as e:
                self.get_logger().warning(f'Invalid reference path: {e}')

        def publish(self,v,delta):
            wheel_speed=float(v)/args.wheel_radius
            drive=Float64MultiArray()
            drive.data=[args.drive_left_sign*wheel_speed,
                        args.drive_right_sign*wheel_speed]
            steer=Float64MultiArray()
            steer.data=[args.steering_left_sign*float(delta),
                        args.steering_right_sign*float(delta)]
            self.drive_pub.publish(drive)
            self.steer_pub.publish(steer)

        def control_cycle(self):
            now=self.get_clock().now().nanoseconds*1e-9
            if (self.observation is None or self.last_odom_stamp is None or
                now-self.last_odom_stamp > args.odom_timeout):
                self.publish(0.,0.)
                if now-self.last_warning_sec>2.:
                    self.get_logger().warning('No fresh odometry; publishing STOP')
                    self.last_warning_sec=now
                return
            # Both ROS modes require a genuine external path before driving.
            # build(args) constructs a synthetic path for standalone simulation;
            # never accidentally command Gazebo using that startup path.
            if self.last_path_stamp is None:
                self.publish(0., 0.)
                if now - self.last_warning_sec > 2.:
                    self.get_logger().warning(
                        'Waiting for valid external reference path; publishing STOP')
                    self.last_warning_sec = now
                return
            # Lane mode must NOT follow old camera geometry when perception is stale.
            if args.input_mode == 'lane' and (
                self.last_path_stamp is None or
                now - self.last_path_stamp > args.path_timeout
            ):
                self.publish(0., 0.)
                if now - self.last_warning_sec > 2.:
                    self.get_logger().warning(
                        'No fresh camera-derived nav_msgs/Path; publishing STOP')
                    self.last_warning_sec = now
                return
            if (args.input_mode == 'lane' and args.ctrl_mode == 'MPC' and
                    args.lane_mpc_finite_path and self.accepted_path_xy is not None):
                remaining = lane_remaining_preview_m(self.accepted_path_xy,
                                                     self.observation[:2])
                if remaining < args.lane_mpc_min_remaining_preview_m:
                    self.publish(0., 0.)
                    if now - self._preview_stop_logged > 2.:
                        self.get_logger().warning(
                            'Insufficient observed camera preview ({:.3f}m < {:.3f}m); STOP'.format(
                                remaining, args.lane_mpc_min_remaining_preview_m))
                        self._preview_stop_logged = now
                    return
            if self.start_control_stamp is None:
                self.start_control_stamp=now
            t=now-self.start_control_stamp
            if self.last_control_stamp is not None and t<self.last_control_stamp:
                my_ctrl.reset()
                self.start_control_stamp=now
                t=0.
            self.last_control_stamp=t
            observation=self.observation.copy()
            try:
                action=my_ctrl.compute_action(t,observation)
            except Exception as e:
                self.get_logger().error(f'Controller failure: {e}')
                self.publish(0.,0.)
                return
            self.publish(float(action[0]),float(action[1]))
            # The sampled controller always computes its own metrics from observation.
            if my_ctrl.last_sample_computed:
                logger.log(t,observation,action,my_ctrl)
                if live_animator is not None:
                    live_animator.record(t, observation, action, my_ctrl)

        def destroy_node(self):
            self.publish(0.,0.)
            try:
                if self._diag_file is not None:
                    self._diag_file.close()
            finally:
                return super().destroy_node()

    rclpy.init()
    node=ROSPreset()
    if not args.animate:
        try:
            rclpy.spin(node)  # one thread, one timer, no rcognita simulator stepping
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    else:
        # Matplotlib GUI frameworks require plt.show() on the main thread.
        # ROS callbacks (including the one and ONLY compute_action call site)
        # run in one SingleThreadedExecutor on a worker thread.
        import threading
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor_thread = threading.Thread(target=executor.spin,
                                           name='qcar_ros_executor', daemon=True)
        executor_thread.start()
        try:
            live_animator.show_live(refresh_hz=args.gui_refresh_hz)
        except KeyboardInterrupt:
            pass
        finally:
            # Stop ROS callbacks before shutting down the node, then publish
            # zero wheel and steering commands on exit.
            executor.shutdown(timeout_sec=2.0)
            executor_thread.join(timeout=2.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__=='__main__':
    main()
