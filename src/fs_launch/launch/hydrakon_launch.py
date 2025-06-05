"""
Hydrakon Formula Student Launch File
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    
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
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.25',
        description='YOLO confidence threshold'
    )
    
    iou_threshold_arg = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.7',
        description='YOLO IoU threshold'
    )
    
    enable_foxglove_arg = DeclareLaunchArgument(
        'enable_foxglove',
        default_value='true',
        description='Enable Foxglove Bridge'
    )
    
    enable_rosbridge_arg = DeclareLaunchArgument(
        'enable_rosbridge',
        default_value='false',
        description='Enable ROS Bridge for web interface'
    )
    
    model_path = LaunchConfiguration('model_path')
    foxglove_port = LaunchConfiguration('foxglove_port')
    camera_fps = LaunchConfiguration('camera_fps')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    iou_threshold = LaunchConfiguration('iou_threshold')
    
    camera_detection_node = Node(
        package='fs_camera',
        executable='camera_detection_node',
        name='hydrakon_vcone_tracker',
        output='screen',
        parameters=[{
            'model_path': model_path,
            'camera_fps': camera_fps,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
        }]
    )
    
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': foxglove_port,
            'address': '0.0.0.0',
            'num_threads': 2,
            'send_buffer_limit': 5000000,
            'max_update_ms': 100,
            'use_compression': True,
            'compression_level': 9,
            'topic_whitelist': [
                '/zed2i/detections_data',
                '/zed2i/cone_detections',
                '/tf_static'
            ],
        }],
        condition=IfCondition(LaunchConfiguration('enable_foxglove'))
    )
    
    rosbridge_server_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{
            'port': 9090,
            'address': '0.0.0.0',
            'max_message_size': 10000000,
            'fragment_timeout': 600,
            'delay_between_messages': 0,
        }],
        condition=IfCondition(LaunchConfiguration('enable_rosbridge'))
    )
    
    # Static transform: base_link to camera
    static_transform_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_transform',
        arguments=['0.5', '0', '1.2', '0', '0.1', '0', 'base_link', 'zed2i_left_camera_frame'],
        output='screen'
    )
    
    # Get IP address for connection info
    get_ip_process = ExecuteProcess(
        cmd=['bash', '-c', 'echo "Foxglove: ws://$(hostname -I | awk \'{print $1}\'):8765"'],
        output='screen'
    )
    
    # Network optimization info
    network_info_process = ExecuteProcess(
        cmd=['bash', '-c', 'echo "ULTRA Mode: 320x180@7.5fps detection + data only"'],
        output='screen'
    )
    
    return LaunchDescription([
        model_path_arg,
        foxglove_port_arg,
        camera_fps_arg,
        confidence_threshold_arg,
        iou_threshold_arg,
        enable_foxglove_arg,
        enable_rosbridge_arg,
        
        LogInfo(msg="HYDRAKON FORMULA STUDENT"),
        get_ip_process,
        network_info_process,
        LogInfo(msg="=" * 60),
        
        camera_detection_node,
        static_transform_camera,
        
        foxglove_bridge_node,
        rosbridge_server_node,
        
        LogInfo(msg="✅ All systems online"),
    ])