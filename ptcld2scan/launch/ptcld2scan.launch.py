from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ptcld2scan',  # Replace with the actual package name
            executable='projector',
            name='projector',
            parameters=['config.yaml']  # Load parameters from the config file
        ),
        Node(
           package='tf2_ros',
           executable='static_transform_publisher',
           name='test',
           arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'depth_link']
        )
    ])

