import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    from ament_index_python.packages import get_package_share_directory
    package_dir = get_package_share_directory('scan2cost')

    config = os.path.join(package_dir, 'config', 'config.yaml')

    return LaunchDescription([
        Node(
            package='scan2cost',
            executable='scan2cost',
            name='scan2cost',
            output='screen',
            parameters=[config]
        )
    ])
