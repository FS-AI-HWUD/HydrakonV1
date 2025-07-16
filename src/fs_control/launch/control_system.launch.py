from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get control module directory
    control_pkg_dir = FindPackageShare('fs_control')
    
    # Config file path - SINGLE SOURCE OF TRUTH
    pid_config = PathJoinSubstitution([control_pkg_dir, 'config', 'pid_params.yaml'])
    
    return LaunchDescription([
        # PID Controller Node
        Node(
            package='fs_control',
            executable='pid_controller',
            name='pid_controller',
            output='screen',
            parameters=[pid_config]
        ),
        
        # Vehicle Interface Node
        Node(
            package='fs_control',
            executable='vehicle_interface',
            name='vehicle_interface',
            output='screen',
            parameters=[pid_config]
        ),
    ])