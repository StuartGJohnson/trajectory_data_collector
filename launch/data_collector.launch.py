from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value="data_collector_sim.yaml",
            description="Config file name or path"
        ),

        Node(
            package="trajectory_data_collector",
            executable="data_collector.py",
            arguments=["--config", LaunchConfiguration("config")],
            output="screen",
        ),
    ])
