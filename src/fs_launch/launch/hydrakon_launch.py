"""
Hydrakon Formula Student Launch File
Launches camera detection and Foxglove streaming for the car
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/dalek/Documents/Zed_2i/cone/best_trt_fp16_640.engine',
        description='Path to TensorRT cone detection model'
    )
    
    foxglove_port_arg = DeclareLaunchArgument(
        'foxglove_port',
        default_value='8765',
        description='Foxglove WebSocket port'
    )
    
    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',
        default_value='30',
        description='Camera capture FPS'
    )
    
    stream_fps_arg = DeclareLaunchArgument(
        'stream_fps',
        default_value='15',
        description='Streaming FPS to reduce bandwidth'
    )
    
    image_quality_arg = DeclareLaunchArgument(
        'image_quality',
        default_value='80',
        description='JPEG compression quality (1-100)'
    )
    
    # Get launch configurations
    model_path = LaunchConfiguration('model_path')
    foxglove_port = LaunchConfiguration('foxglove_port')
    camera_fps = LaunchConfiguration('camera_fps')
    stream_fps = LaunchConfiguration('stream_fps')
    image_quality = LaunchConfiguration('image_quality')
    
    # Camera detection node
    camera_detection_node = Node(
        package='fs_camera',
        executable='camera_detection_node',
        name='hydrakon_camera_detection',
        output='screen',
        parameters=[{
            'model_path': model_path,
            'camera_fps': camera_fps,
            'stream_fps': stream_fps,
            'image_quality': image_quality,
            'detection_confidence': 0.25,
            'iou_threshold': 0.7,
        }]
    )
    
    # Foxglove Bridge for streaming
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': foxglove_port,
            'address': '0.0.0.0',
            'num_threads': 4,
        }]
    )
    
    # Static transform: base_link to camera
    static_transform_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_transform',
        arguments=['0.5', '0', '1.2', '0', '0.1', '0', 'base_link', 'zed2i_left_camera_frame']
    )
    
    return LaunchDescription([
        # Launch arguments
        model_path_arg,
        foxglove_port_arg,
        camera_fps_arg,
        stream_fps_arg,
        image_quality_arg,
        
        # Nodes
        camera_detection_node,
        foxglove_bridge_node,
        static_transform_camera,
    ])