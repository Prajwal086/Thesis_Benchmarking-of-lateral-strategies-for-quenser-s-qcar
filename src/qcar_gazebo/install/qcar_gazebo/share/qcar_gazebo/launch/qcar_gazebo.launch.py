import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg = get_package_share_directory('qcar_gazebo')
    realsense_pkg = get_package_share_directory('realsense2_description')
    xacro_file = os.path.join(pkg, 'urdf', 'qcar_model.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()
    
    install_share = os.path.realpath(os.path.join(pkg,'..'))
    #realsense_share = '/opt/ros/humble/share'
    
    model_path = ':'.join([
        os.path.join(pkg, 'models'),
        install_share,
        #realsense_share,
        #os.path.join(realsense_share, 'meshes'),
        os.environ.get('GAZEBO_MODEL_PATH', '')
    ])

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose',
                 os.path.join(pkg, 'worlds', 'empty.world'),
                 '-s', 'libgazebo_ros_factory.so'],
            additional_env={
                'GAZEBO_MODEL_DATABASE_URI': '',
                'GAZEBO_MODEL_PATH': model_path,
            },
            output='screen'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'qcar'],
            output='screen'
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            output='screen'
        ),
        Node(
	    package='controller_manager',
	    executable='spawner',
	    arguments=['joint_state_broadcaster'],
	    output='screen'
	),
	Node(
	    package='controller_manager',
	    executable='spawner',
	    arguments=['drive_controller'],
	    output='screen'
	),
	Node(
	    package='controller_manager',
	    executable='spawner',
	    arguments=['steering_controller'],
	    output='screen'
	),
    ])
