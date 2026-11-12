# QCar front CSI camera — intrinsic calibration

**Camera stream resolution:** 820 x 410
**Target:** 8 x 6 inner chessboard corners, displayed on a flat phone screen
**Model:** OpenCV pinhole + 5-term Brown–Conrady distortion (`plumb_bob`)
**Motor actuation:** keep DISABLED until camera mounting, state estimation, and the complete command path are independently validated.

## Final parameters
- `camera_fx`: 398.778234173 pixels
- `camera_fy`: 265.339759099 pixels
- `camera_cx`: 449.890537077 pixels
- `camera_cy`: 197.216777426 pixels
- distortion `(k1,k2,p1,p2,k3)`: -0.336792620,0.142543525,0.000815627,-0.001322747,-0.032608145
- fit RMS: 0.2469 pixels (48 selected calibration views)

## Data quality
- Original photos: 30; latest photos: 28; **58 total**.
- Full 8x6 pattern detected by `cv2.findChessboardCornersSB`: **50**.
- 2 views rejected for large initial per-view reprojection errors:
  `new_view_001` (~1.404 px) and `new_view_014` (~0.853 px).
- Final fit: **48 selected views; RMS 0.2469 px**.
- Independent previous-vs-new photo calibrations agree reasonably well on focal length and principal point.
- Four parameter-stability checks and five holdout reprojection checks were performed.
- CAUTION: Chessboard corner coverage reaches approximately x=192..733, y=24..265; the lower part of the image is poorly sampled.
  Calibrated intrinsics are reasonable, but bottom-of-image undistortion must be validated using physical scene geometry.
- Camera pose relative to robot (`height`, `pitch`, `lateral offset`, `forward offset`) is NOT calibrated here.
- Gyroscope/encoder odometry is NOT independent ground-truth positioning.
- Changing resolution, cropping, camera lens, or camera stream processing invalidates or requires rescaling/rechecking this calibration.

## Files
- `qcar_relay_camera_params.yaml`: **USE THIS** directly for your generated laptop relay.
- `qcar_front_camera_calibration.yaml`: conventional ROS raw camera YAML for archive/other tools; the current relay does not load this one directly.
- `camera_parameters.json`: complete numeric calibration and stability checks.
- `per_image_calibration_checks.csv`: detected/accepted views and error values.
- `raw_vs_undistorted_example.jpg`: visual review.
- `previous_photos_contact.jpg`, `new_photos_contact.jpg`: image coverage overview.

## Step 1: copy YAML beside relay (laptop)
Copy `qcar_relay_camera_params.yaml` onto your laptop, for example:

```bash
cp qcar_relay_camera_params.yaml ~/qcar_relay_camera_params.yaml
```

## Step 2: restart laptop relay (motors DISABLED)
Leave the QCar-side TCP bridge running. Stop the old laptop relay first. Then:

```bash
python3 pc_qcar_offboard_relay.py \
  --ros-args \
  --params-file ~/qcar_relay_camera_params.yaml \
  -p qcar_ip:=YOUR_QCAR_IP \
  -p enable_rcognita_interface:=true \
  -p enable_imu_odometry:=true \
  -p gyro_axis_validated:=true \
  -p gyro_yaw_axis:=2 \
  -p gyro_yaw_sign:=1.0 \
  -p gyro_rad_per_unit:=1.0 \
  -p gyro_stationary_bias:=0.004431500053891852 \
  -p enable_hardware_commands:=false
```

**Safety:** `gyro_axis_validated:=true` and `gyro_yaw_sign:=1.0` above assume your manual counterclockwise rotation test CONFIRMED that the QCar Z gyro becomes positive. If this has not been confirmed, do not enable IMU odometry yet. Do not enable hardware commands for camera tests.

## Step 3: verify camera and odometry topics

```bash
ros2 param get /pc_qcar_offboard_relay camera_fx
ros2 topic echo /front_camera/camera_info --once
ros2 topic hz /front_camera/image_raw
ros2 topic echo /odom/bot --once
```

Expected CameraInfo:
- `width=820`, `height=410`
- `K` contains the calibrated `fx,fy,cx,cy` values above.
- As the relay is **rectifying frames** via OpenCV, the **published rectified CameraInfo has D=[0,0,0,0,0]**; this is CORRECT and intentional.
- The *raw* calibration YAML contains the nonzero distortion coefficients used by the relay for rectification.

## Step 4: run the UNCHANGED rcognita lane detector (commands still disabled)

```bash
python3 presets/lane_detection.py \
  --ros-args \
  -p use_sim_time:=false \
  -p image_topic:=/front_camera/image_raw \
  -p camera_info_topic:=/front_camera/camera_info \
  -p odom_topic:=/odom/bot \
  -p lane_path_topic:=/lane_detection/reference_path \
  -p require_both_boundaries:=true \
  -p require_camera_info_for_path:=true \
  -p right_white_fallback:=true \
  -p min_lane_status_for_path:=3 \
  -p debug:=true
```

Verify in a separate terminal:

```bash
ros2 topic echo /front_camera/camera_info --once
ros2 topic echo /lane_detection/status --once
ros2 topic hz /lane_detection/reference_path
```

The missing-camera-calibration warning should disappear after valid `CameraInfo` messages arrive.
Valid camera calibration does **not** guarantee a lane reference path: the detector can still withhold the path when boundaries or metric path geometry fail its quality checks.

## Before any physical autonomous driving
1. Confirm camera stream orientation, calibration, distortion correction, and metric ground-plane mounting parameters against a known physical distance.
2. Validate IMU heading sign, stationary drift, encoder polarity/scale, odometry frame, and the rear-axle offset.
3. Confirm actuator steering sign, drive speed command conversion, stale-data safety, and emergency stop at zero throttle with the robot secured.
4. Test offboard compatibility while `enable_hardware_commands:=false` first; do not enable autonomous motion on the basis of camera calibration alone.
