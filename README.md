# QCar ROS 2 workspace

This workspace contains a Gazebo Classic QCar simulation package and Python
scripts for lane detection, reference-path following, and standalone controller
simulation.

## Repository layout

```text
ros2_ws/
├── src/
│   ├── qcar_gazebo/       # ROS 2 / Gazebo Classic simulation package
│   └── rcognita/          # Controller and perception Python source
│       └── presets/
└── README.md
```

`qcar_gazebo` is built and installed by `colcon`. The scripts under
`src/rcognita/presets` are run directly with Python; they are not installed as
ROS executables by this workspace.

## Prerequisites

- Ubuntu with a ROS 2 distribution and a compatible **Gazebo Classic**
  installation. The launch file invokes `gazebo`, not modern Gazebo (`gz`).
- ROS 2 packages used by the simulator, including `gazebo_ros`,
  `robot_state_publisher`, `controller_manager`, `rviz2`, `xacro`, and
  `realsense2_description`.
- `colcon`, `rosdep`, and the ROS Python packages used by the presets, including
  `rclpy`, `cv_bridge`, and the ROS message packages.
- Python dependencies for `rcognita`, including NumPy, SciPy, and Matplotlib.
  Use the system Python environment sourced for ROS 2 so Python can import ROS
  packages. A virtual environment that cannot see the ROS Python installation
  will not run the ROS scripts.

The `qcar_gazebo/package.xml` declares only part of the launch-time dependency
set. Check that the Gazebo Classic and ROS 2 packages above are installed for
your ROS distribution; `rosdep` may not discover dependencies that are not
declared there.

## Build the workspace

From a terminal, set `ROS_DISTRO` to the installed ROS 2 distribution name
(for example, `humble`), then build:

```bash
cd ~/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Source the ROS and workspace setup files in every new terminal before running
ROS commands:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/ros2_ws
source install/setup.bash
cd src/rcognita
```

The commands below assume the current directory is
`~/ros2_ws/src/rcognita`, unless otherwise noted.

## Gazebo lane-following simulation

Start Gazebo in one terminal. The `world` argument accepts `circle`, `s_curve`,
`figure_eight`, or `empty`:

```bash
cd ~/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
ros2 launch qcar_gazebo qcar_gazebo.launch.py world:=circle
```

In another terminal, start the lane detector:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/ros2_ws
source install/setup.bash
cd src/rcognita
python3 presets/lane_detection.py \
  --ros-args \
  -p use_sim_time:=false \
  -p debug:=true
```

Then start a controller in a third terminal. Example using Pure Pursuit:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/ros2_ws
source install/setup.bash
cd src/rcognita
MPLBACKEND=TkAgg python3 presets/main_qcar_ros.py \
  --ctrl_mode PP \
  --input_mode lane \
  --lane_path_topic /lane_detection/reference_path \
  --odom_twist_frame magnitude \
  --target_speed 0.5 \
  --speed_policy adaptive \
  --csv PP_c_200ms.csv \
  --animate \
  --ros-args -p use_sim_time:=false
```

Example using MPC:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/ros2_ws
source install/setup.bash
cd src/rcognita
MPLBACKEND=TkAgg python3 presets/main_qcar_ros.py \
  --ctrl_mode MPC \
  --input_mode lane \
  --odom_topic /odom/bot \
  --odom_twist_frame magnitude \
  --lane_path_topic /lane_detection/reference_path \
  --target_speed 0.5 \
  --speed_policy adaptive \
  --max_lateral_acceleration 0.4 \
  --dt 0.10 \
  --Nactor 15 \
  --pred_step_size_multiplier 3 \
  --optimizer_max_iterations 8 \
  --csv MPC_c_100ms.csv \
  --animate \
  --ros-args -p use_sim_time:=false
```

`--animate` opens a GUI and requires a graphical display and Tk support. For a
headless run, omit `--animate` and `MPLBACKEND=TkAgg`. The `magnitude` odometry
twist option uses speed magnitude and does not preserve forward/reverse
direction; use it only for forward-driving experiments.

## Gazebo reference-trajectory following

Start Gazebo as described above. Then start the reference-path publisher in a
separate terminal:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/ros2_ws
source install/setup.bash
cd src/rcognita
python3 -u presets/reference_path_publisher.py \
  --ros-args -p use_sim_time:=false -p path_type:=figure_eight
```

In another terminal, run the ROS controller:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/ros2_ws
source install/setup.bash
cd src/rcognita
MPLBACKEND=TkAgg python3 presets/main_qcar_ros.py \
  --ctrl_mode PP \
  --input_mode path \
  --target_speed 0.5 \
  --speed_policy adaptive \
  --car_display_mode icon \
  --csv PP.csv \
  --animate \
  --ros-args -p use_sim_time:=false
```

The controller listens on `/reference_path` in path mode. The publisher's
`path_type` and other geometry parameters can be overridden with ROS parameters;
see `presets/reference_path_publisher.py`.

## Physical QCar (hardware)

Hardware operation depends on the separate `qcar_hardware_control` repository
and a laptop-to-QCar network connection. That hardware repository and its
scripts are not included in this workspace. The commands below follow the
uploaded setup; run them only after verifying the configured topics, sensor
frames, wheel signs, and actuator behavior for your vehicle.

1. On the QCar, start its offboard bridge using the hardware repository's
   instructions. The uploaded command was:

   ```bash
   cd ~/ros_2_Prajwal/src/qcar_hardware_control/qcar_hardware_control
   python3 qcar_offboard_bridge.py
   ```

2. On the laptop, start the relay. Replace `YOUR_QCAR_IP` with the QCar's
   reachable IP address. Keep hardware commands explicitly disabled during
   camera and odometry validation:

   ```bash
   python3 pc_qcar_offboard_relay.py \
     --ros-args \
     --params-file ~/ros2_ws/src/rcognita/QCar_front_camera_calibration_final/qcar_relay_camera_params.yaml \
     -p qcar_ip:=YOUR_QCAR_IP \
     -p enable_hardware_commands:=false
   ```

3. In `~/ros2_ws/src/rcognita`, run the lane detector:

   ```bash
   python3 presets/lane_detection.py \
     --ros-args \
     -p use_sim_time:=false \
     -p debug:=true
   ```

4. Verify camera and odometry data in another terminal:

   ```bash
   ros2 topic echo /front_camera/camera_info --once
   ros2 topic hz /front_camera/image_raw
   ros2 topic echo /odom/bot --once
   ros2 topic echo /lane_detection/status --once
   ros2 topic hz /lane_detection/reference_path
   ```

5. Only after the independent hardware checks are complete, start the lane
   controller. The example uses the uploaded Pure Pursuit settings:

   ```bash
   source /opt/ros/$ROS_DISTRO/setup.bash
   cd ~/ros2_ws
   source install/setup.bash
   cd src/rcognita
   MPLBACKEND=TkAgg python3 presets/main_qcar_ros.py \
     --ctrl_mode PP \
     --input_mode lane \
     --lane_path_topic /lane_detection/reference_path \
     --odom_twist_frame magnitude \
     --target_speed 0.5 \
     --speed_policy adaptive \
     --csv PP_c_200ms.csv \
     --animate \
     --ros-args -p use_sim_time:=false
   ```

**Hardware safety:** Camera calibration alone does not calibrate camera pose,
verify odometry, or validate steering and drive signs. Follow
[`README_FIRST.md`](src/rcognita/QCar_front_camera_calibration_final/README_FIRST.md)
and verify the complete command path with the vehicle secured and hardware
commands disabled before enabling motion. Keep an accessible emergency stop.
Do not treat the sample parameters or topic conventions as validated for a
different QCar installation.

## Standalone controller simulation

These runs do not require ROS nodes or Gazebo, but still need the Python
dependencies listed above. From `~/ros2_ws/src/rcognita`:

Pure Pursuit on a circle:

```bash
python3 presets/main_qcar.py \
  --ctrl_mode PP \
  --path_type circle \
  --t1 30 \
  --target_speed 1.0 \
  --speed_policy adaptive \
  --csv pp_circle_1.0.csv \
  --animate
```

MPC on a figure-eight:

```bash
python3 presets/main_qcar.py \
  --ctrl_mode MPC \
  --path_type figure_eight \
  --Nactor 15 \
  --pred_step_size_multiplier 1 \
  --t1 30 \
  --speed_policy adaptive \
  --target_speed 0.5 \
  --max_lateral_acceleration 0.25 \
  --csv mpc_fig_8_adaptive.csv \
  --dashboard_png mpc_eight_dashboard.png \
  --save_animation mpc_fig_8.gif \
  --animate
```

The standalone preset also accepts `--ctrl_mode PID` and `--ctrl_mode Stanley`.
For figure-eight runs, choose a `--start_index` away from the path crossing if
the initial projection is ambiguous. Use `--help` for the full list and
descriptions of controller and perception parameters:

```bash
python3 presets/main_qcar.py --help
python3 presets/main_qcar_ros.py --help
```

## Git and generated files

Commit the source packages, configuration, launch files, worlds, models,
presets, and documentation. Do not commit generated ROS/Python build outputs
(`build/`, `install/`, `log/`), Python caches, or run outputs such as CSV,
PNG, and GIF results. Keep machine-specific IP addresses, credentials, and
local paths out of committed configuration.
