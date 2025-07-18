"""
Hydrakon Formula Student Launch File
Enhanced with Robosense LiDAR Integration, FS Control Module, and Planning Node
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/dalek/Documents/Zed_2i/cone/best_trt_fp16_640.engine',
        description='Path to TensorRT cone detection model'
    )
    
    lidar_config_path_arg = DeclareLaunchArgument(
        'lidar_config_path',
        default_value='/home/dalek/ros2_ws/src/rslidar_sdk/config/my_config.yaml',
        description='Path to Robosense LiDAR configuration file'
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
    
    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar',
        default_value='true',
        description='Enable Robosense LiDAR'
    )

    enable_clustering_arg = DeclareLaunchArgument(
        'enable_clustering',
        default_value='true',
        description='Enable LiDAR cone clustering'
    )
    
    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz',
        default_value='false',
        description='Launch RViz2 for visualization'
    )

    enable_gps_arg = DeclareLaunchArgument(
        'enable_gps',
        default_value='true',
        description='Enable CHCNAV INS GPS bridge'
    )
    
    enable_control_arg = DeclareLaunchArgument(
        'enable_control',
        default_value='true',
        description='Enable FS Control Module'
    )
    
    # NEW: Planning node arguments
    enable_planning_arg = DeclareLaunchArgument(
        'enable_planning',
        default_value='true',
        description='Enable Hydrakon Planning Node'
    )
    
    target_laps_arg = DeclareLaunchArgument(
        'target_laps',
        default_value='10',
        description='Target number of laps (set to 0 for unlimited)'
    )
    
    min_lap_time_arg = DeclareLaunchArgument(
        'min_lap_time',
        default_value='150.0',
        description='Minimum lap time in seconds for valid lap counting'
    )
    
    min_speed_arg = DeclareLaunchArgument(
        'min_speed',
        default_value='2.0',
        description='Minimum vehicle speed in m/s'
    )
    
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='3.0',
        description='Maximum vehicle speed in m/s'
    )
    
    orange_gate_threshold_arg = DeclareLaunchArgument(
        'orange_gate_threshold',
        default_value='2.0',
        description='Distance threshold for orange gate passage in meters'
    )
    
    orange_cooldown_arg = DeclareLaunchArgument(
        'orange_cooldown',
        default_value='3.0',
        description='Cooldown between orange gate detections in seconds'
    )
    
    model_path = LaunchConfiguration('model_path')
    lidar_config_path = LaunchConfiguration('lidar_config_path')
    foxglove_port = LaunchConfiguration('foxglove_port')
    camera_fps = LaunchConfiguration('camera_fps')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    iou_threshold = LaunchConfiguration('iou_threshold')
    
    # === PERCEPTION NODES ===
    
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
    
    lidar_node = Node(
        package='rslidar_sdk',
        executable='rslidar_sdk_node',
        name='rslidar_sdk_node',
        output='screen',
        parameters=[{
            'config_path': lidar_config_path,
        }],
        condition=IfCondition(LaunchConfiguration('enable_lidar'))
    )

    lidar_cluster_node = Node(
        package='fs_lidar',
        executable='rslidar_cluster',
        name='rslidar_cluster',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_clustering'))
    )

    nmea_gps_bridge_node = Node(
        package='fs_planning',
        executable='nmea_bridge',
        name='chcnav_ins_bridge',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_gps'))
    )
    
    # === PLANNING NODE ===
    
    planning_node = Node(
        package='fs_planning',
        executable='hydrakon_planning',
        name='hydrakon_planning_node',
        output='screen',
        parameters=[{
            'target_laps': LaunchConfiguration('target_laps'),
            'min_lap_time': LaunchConfiguration('min_lap_time'),
            'min_speed': LaunchConfiguration('min_speed'),
            'max_speed': LaunchConfiguration('max_speed'),
            'orange_gate_threshold': LaunchConfiguration('orange_gate_threshold'),
            'orange_cooldown': LaunchConfiguration('orange_cooldown'),
        }],
        condition=IfCondition(LaunchConfiguration('enable_planning'))
    )
    
    # === FS CONTROL MODULE ===
    
    # Get control module directory for config
    control_pkg_dir = FindPackageShare('fs_control')
    pid_config = PathJoinSubstitution([control_pkg_dir, 'config', 'pid_params.yaml'])
    
    # PID Controller Node
    pid_controller_node = Node(
        package='fs_control',
        executable='pid_controller',
        name='pid_controller',
        output='screen',
        parameters=[pid_config],
        condition=IfCondition(LaunchConfiguration('enable_control'))
    )
    
    # Vehicle Interface Node
    vehicle_interface_node = Node(
        package='fs_control',
        executable='vehicle_interface',
        name='vehicle_interface',
        output='screen',
        parameters=[pid_config],
        condition=IfCondition(LaunchConfiguration('enable_control'))
    )
    
    # === VISUALIZATION & MONITORING ===
    
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
                '/zed2i/raw_feed',
                '/lidar/points',
                '/perception/lidar_cluster',
                '/perception/cone_markers',
                '/ins/gnss',
                '/ins/nav',
                '/ins/heading',
                '/ins/velocity',
                '/tf',
                '/tf_static',
                '/acceleration_cmd',
                '/planning/reference_steering',
                '/hydrakon_can/command',
                '/current_speed',
                '/imu/data',
                '/cmd_vel',
                '/planning_stats'
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
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_rviz'))
    )
    
    # === STATIC TRANSFORMS ===
    
    # Static transform: map to base_link
    static_transform_map_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_base_transform',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
        output='screen'
    )
    
    # Static transform: base_link to rslidar (LiDAR mounting position)
    static_transform_base_to_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_lidar_transform',
        # Need to adjust these values based on actual LiDAR mounting position, added placeholder values for now
        # Format: x y z roll pitch yaw parent_frame child_frame
        arguments=['0', '0', '1.5', '0', '0', '0', 'base_link', 'rslidar'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_lidar'))
    )
    
    # Static transform: base_link to camera
    static_transform_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_transform',
        arguments=['0.5', '0', '1.2', '0', '0.1', '0', 'base_link', 'zed2i_left_camera_frame'],
        output='screen'
    )
    
    static_transform_base_to_gps = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_gps_transform',
        arguments=['0', '0', '2.0', '0', '0', '0', 'base_link', 'gps'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_gps'))
    )
    
    # === LAUNCH INFORMATION ===
    
    get_ip_process = ExecuteProcess(
        cmd=['bash', '-c', 'echo "Foxglove: ws://$(hostname -I | awk \'{print $1}\'):8765"'],
        output='screen'
    )
    
    lidar_info_process = ExecuteProcess(
        cmd=['bash', '-c', 'echo "LiDAR: Robosense Helios 16 @ 192.168.1.200"'],
        output='screen'
    )
    
    gps_info_process = ExecuteProcess(
        cmd=['bash', '-c', 'echo "GPS/INS: CHCNAV CGI-410 @ 192.168.1.201"'],
        output='screen'
    )
    
    control_info_process = ExecuteProcess(
        cmd=['bash', '-c', 'echo "Control: PID Controller + Vehicle Interface → ADS-DV CAN"'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_control'))
    )
    
    planning_info_process = ExecuteProcess(
        cmd=['bash', '-c', 'echo "Planning: Pure Pursuit + Lap Counter → /cmd_vel"'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_planning'))
    )
    
    network_info_process = ExecuteProcess(
        cmd=['bash', '-c', 'echo "Systems Active: LiDAR, Camera, GPS/INS, Planning, Control, Foxglove"'],
        output='screen'
    )
    
    return LaunchDescription([
        model_path_arg,
        lidar_config_path_arg,
        foxglove_port_arg,
        camera_fps_arg,
        confidence_threshold_arg,
        iou_threshold_arg,
        enable_foxglove_arg,
        enable_rosbridge_arg,
        enable_lidar_arg,
        enable_clustering_arg,
        enable_rviz_arg,
        enable_gps_arg,
        enable_control_arg,
        enable_planning_arg,
        target_laps_arg,
        min_lap_time_arg,
        min_speed_arg,
        max_speed_arg,
        orange_gate_threshold_arg,
        orange_cooldown_arg,
        
        LogInfo(msg="HYDRAKON FS-AI SYSTEM WITH PLANNING"),
        LogInfo(msg="=" * 60),
        get_ip_process,
        lidar_info_process,
        gps_info_process,
        control_info_process,
        planning_info_process,
        network_info_process,
        LogInfo(msg="=" * 60),
        
        # Perception
        camera_detection_node,
        # lidar_node,
        # lidar_cluster_node,
        # nmea_gps_bridge_node,
        
        # Planning
        planning_node,
        
        # Control System
        # speed_processor_node,
        # pid_controller_node,
        # vehicle_interface_node,
        
        # Transforms
        # static_transform_map_to_base,
        # static_transform_base_to_lidar,
        # static_transform_base_to_camera,
        # static_transform_base_to_gps,
        
        # Visualization
        foxglove_bridge_node,
        rosbridge_server_node,
        # rviz_node,
        
        LogInfo(msg="✅ All systems online!"),
    ])