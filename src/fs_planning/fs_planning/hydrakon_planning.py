#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String, Bool

class CombinedController(Node):
    def __init__(self):
        super().__init__('combined_controller')
        
        # Subscribe to camera detections
        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            '/zed2i/detections_data',
            self.detection_callback,
            10
        )
        
        # Subscribe to CAN state
        self.state_subscription = self.create_subscription(
            String,
            '/hydrakon_can/state_str',
            self.state_callback,
            10
        )
        
        # Publishers
        self.command_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/hydrakon_can/command',
            10
        )
        
        self.driving_flag_publisher = self.create_publisher(
            Bool,
            '/hydrakon_can/driving_flag',
            10
        )
        
        # Timer to publish driving flag at 20Hz (50ms)
        self.timer = self.create_timer(0.05, self.publish_driving_flag)
        
        # Camera parameters
        self.image_width = 1280
        self.image_height = 720
        self.image_center_x = self.image_width // 2
        
        # Steering parameters
        self.max_steering_angle = 0.4
        self.steering_gain = 0.003
        self.min_cone_distance = 50
        
        # Cone class IDs
        self.YELLOW_CONE = 0
        self.BLUE_CONE = 1
        
        # Default drive parameters
        self.default_speed = 0.0
        self.default_acceleration = 0.0
        self.cone_pair_acceleration = 0.9  # Acceleration when both cone types are detected
        
        # State tracking for driving flag
        self.current_as_state = None
        self.current_ami_state = None
        
        # Last steering angle for continuity
        self.last_steering_angle = 0.0
        
        self.get_logger().info('Combined Controller initialized')
        self.get_logger().info('- Camera-based steering control')
        self.get_logger().info('- AMI state monitoring with 20Hz driving flag')
        self.get_logger().info('- Acceleration when cone pairs detected')
        self.get_logger().info(f'Image dimensions: {self.image_width}x{self.image_height}')
        self.get_logger().info(f'Steering gain: {self.steering_gain}, Max angle: {self.max_steering_angle}')
        self.get_logger().info(f'Cone pair acceleration: {self.cone_pair_acceleration}')
    
    def parse_state_string(self, state_str):
        """Parse the state string to extract AS and AMI values"""
        try:
            as_match = re.search(r'AS:(\w+)', state_str)
            as_state = as_match.group(1) if as_match else None
            
            ami_match = re.search(r'AMI:(\w+)', state_str)
            ami_state = ami_match.group(1) if ami_match else None
            
            return as_state, ami_state
        except Exception as e:
            self.get_logger().error(f'Error parsing state string: {e}')
            return None, None
    
    def state_callback(self, msg):
        """Callback for CAN state updates"""
        as_state, ami_state = self.parse_state_string(msg.data)
        
        if as_state is not None and ami_state is not None:
            self.current_as_state = as_state
            self.current_ami_state = ami_state
            self.get_logger().debug(f'Updated state - AS: {as_state}, AMI: {ami_state}')
    
    def should_set_driving_flag(self):
        """Check if driving flag should be true based on current state"""
        if self.current_as_state is None or self.current_ami_state is None:
            return False
        
        # Check if AS is DRIVING and AMI is not NOT_SELECTED
        return (self.current_as_state == 'DRIVING' and 
                self.current_ami_state != 'NOT_SELECTED')
    
    def publish_driving_flag(self):
        """Publish the driving flag at 20Hz"""
        msg = Bool()
        msg.data = self.should_set_driving_flag()
        self.driving_flag_publisher.publish(msg)
        
        # Log only when state changes to avoid spam
        if hasattr(self, 'last_published_state'):
            if self.last_published_state != msg.data:
                self.get_logger().info(f'Driving flag changed to: {msg.data} (AS: {self.current_as_state}, AMI: {self.current_ami_state})')
        else:
            self.get_logger().info(f'Publishing driving flag: {msg.data} (AS: {self.current_as_state}, AMI: {self.current_ami_state})')
        
        self.last_published_state = msg.data
    
    def detection_callback(self, msg):
        """Process camera detections and calculate steering angle"""
        yellow_cones = []
        blue_cones = []
        
        # Extract cone positions from detections
        for detection in msg.detections:
            if len(detection.results) > 0:
                class_id = int(detection.results[0].hypothesis.class_id)
                confidence = detection.results[0].hypothesis.score
                
                # Get cone center position
                center_x = detection.bbox.center.position.x
                center_y = detection.bbox.center.position.y
                
                # Filter by confidence and classify by color
                if confidence > 0.5:
                    if class_id == self.YELLOW_CONE:
                        yellow_cones.append((center_x, center_y, confidence))
                    elif class_id == self.BLUE_CONE:
                        blue_cones.append((center_x, center_y, confidence))
        
        # Calculate steering angle and acceleration
        steering_angle = self.calculate_steering_angle(yellow_cones, blue_cones)
        acceleration = self.calculate_acceleration(yellow_cones, blue_cones)
        
        # Only publish steering commands if we should be driving
        if self.should_set_driving_flag():
            self.publish_steering_command(steering_angle, acceleration, msg.header.stamp)
            
            # Log detection info
            if len(yellow_cones) > 0 or len(blue_cones) > 0:
                accel_status = "ACCELERATING" if acceleration > 0 else "COASTING"
                self.get_logger().info(f'DRIVING: {len(yellow_cones)} yellow, {len(blue_cones)} blue cones. Steering: {steering_angle:.3f} rad, {accel_status}: {acceleration:.1f}')
        else:
            # Send zero steering and acceleration when not in driving mode
            self.publish_steering_command(0.0, 0.0, msg.header.stamp)
            if len(yellow_cones) > 0 or len(blue_cones) > 0:
                self.get_logger().debug(f'NOT DRIVING: Cones detected but not in driving mode (AS: {self.current_as_state}, AMI: {self.current_ami_state})')
    
    def calculate_acceleration(self, yellow_cones, blue_cones):
        """Calculate acceleration based on cone detection"""
        # Accelerate when both yellow and blue cones are detected (cone pair)
        if len(yellow_cones) > 0 and len(blue_cones) > 0:
            return self.cone_pair_acceleration
        else:
            return self.default_acceleration
    
    def calculate_steering_angle(self, yellow_cones, blue_cones):
        """Calculate steering angle based on cone positions"""
        
        # If no cones detected, return last steering angle for continuity
        if len(yellow_cones) == 0 and len(blue_cones) == 0:
            return self.last_steering_angle * 0.9  # Gradually reduce steering
        
        target_x = None
        
        # Case 1: Both yellow and blue cones detected
        if len(yellow_cones) > 0 and len(blue_cones) > 0:
            closest_yellow = self.find_closest_cone(yellow_cones)
            closest_blue = self.find_closest_cone(blue_cones)
            
            yellow_x = closest_yellow[0]
            blue_x = closest_blue[0]
            
            # Calculate midpoint between closest yellow and blue cones
            target_x = (yellow_x + blue_x) / 2.0
            
            self.get_logger().debug(f'Midpoint steering: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f}, Target: {target_x:.1f}')
        
        # Case 2: Only yellow cones detected
        elif len(yellow_cones) > 0:
            closest_yellow = self.find_closest_cone(yellow_cones)
            yellow_x = closest_yellow[0]
            
            # Target point is to the left of yellow cone
            offset = 100
            target_x = yellow_x - offset
            
            self.get_logger().debug(f'Yellow-only steering: Yellow at {yellow_x:.1f}, Target: {target_x:.1f}')
        
        # Case 3: Only blue cones detected
        elif len(blue_cones) > 0:
            closest_blue = self.find_closest_cone(blue_cones)
            blue_x = closest_blue[0]
            
            # Target point is to the right of blue cone
            offset = 100
            target_x = blue_x + offset
            
            self.get_logger().debug(f'Blue-only steering: Blue at {blue_x:.1f}, Target: {target_x:.1f}')
        
        # Convert target position to steering angle
        if target_x is not None:
            error_pixels = target_x - self.image_center_x
            steering_angle = -error_pixels * self.steering_gain
            steering_angle = max(-self.max_steering_angle, 
                               min(self.max_steering_angle, steering_angle))
            
            self.last_steering_angle = steering_angle
            return steering_angle
        
        return self.last_steering_angle * 0.9
    
    def find_closest_cone(self, cones):
        """Find the closest cone based on y-coordinate"""
        if not cones:
            return None
        
        closest_cone = max(cones, key=lambda cone: cone[1])
        return closest_cone
    
    def publish_steering_command(self, steering_angle, acceleration, timestamp):
        """Publish Ackermann steering command with acceleration"""
        msg = AckermannDriveStamped()
        
        msg.header.stamp = timestamp
        msg.header.frame_id = 'base_link'
        
        msg.drive.steering_angle = float(steering_angle)
        msg.drive.steering_angle_velocity = 0.0
        msg.drive.speed = self.default_speed
        msg.drive.acceleration = float(acceleration)
        msg.drive.jerk = 0.0
        
        self.command_publisher.publish(msg)

def test_mode():
    """Test mode - sends test messages for both driving flag and steering with acceleration"""
    rclpy.init()
    
    node = Node('combined_controller_test')
    
    # Publishers for testing
    command_publisher = node.create_publisher(
        AckermannDriveStamped,
        '/hydrakon_can/command',
        10
    )
    
    driving_flag_publisher = node.create_publisher(
        Bool,
        '/hydrakon_can/driving_flag',
        10
    )
    
    node.get_logger().info('TEST MODE: Combined Controller Test with Acceleration')
    node.get_logger().info('Testing driving flag, steering commands, and acceleration...')
    node.get_logger().info('Press Ctrl+C to stop')
    
    import time
    test_sequence = [
        {'driving': True, 'steering': 0.0, 'acceleration': 0.0},
        {'driving': True, 'steering': 0.1, 'acceleration': 0.9},
        {'driving': True, 'steering': 0.2, 'acceleration': 0.9},
        {'driving': True, 'steering': 0.0, 'acceleration': 0.9},
        {'driving': True, 'steering': -0.1, 'acceleration': 0.0},
        {'driving': True, 'steering': -0.2, 'acceleration': 0.0},
        {'driving': False, 'steering': 0.0, 'acceleration': 0.0},
    ]
    
    sequence_index = 0
    
    try:
        while rclpy.ok():
            test_data = test_sequence[sequence_index % len(test_sequence)]
            
            # Publish driving flag
            flag_msg = Bool()
            flag_msg.data = test_data['driving']
            driving_flag_publisher.publish(flag_msg)
            
            # Publish steering command with acceleration
            cmd_msg = AckermannDriveStamped()
            cmd_msg.header.stamp = node.get_clock().now().to_msg()
            cmd_msg.header.frame_id = 'base_link'
            cmd_msg.drive.steering_angle = float(test_data['steering'])
            cmd_msg.drive.steering_angle_velocity = 0.0
            cmd_msg.drive.speed = 0.0
            cmd_msg.drive.acceleration = float(test_data['acceleration'])
            cmd_msg.drive.jerk = 0.0
            
            command_publisher.publish(cmd_msg)
            
            accel_status = "ACCELERATING" if test_data['acceleration'] > 0 else "COASTING"
            node.get_logger().info(f'Published - Driving: {test_data["driving"]}, Steering: {test_data["steering"]:.1f} rad, {accel_status}: {test_data["acceleration"]:.1f}')
            
            sequence_index += 1
            time.sleep(2.0)
            rclpy.spin_once(node, timeout_sec=0.1)
            
    except KeyboardInterrupt:
        node.get_logger().info('Test mode stopped')
    
    node.destroy_node()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    combined_controller = CombinedController()
    
    try:
        rclpy.spin(combined_controller)
    except KeyboardInterrupt:
        pass
    
    combined_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1 or '--test' in sys.argv:
        print("Running in TEST MODE - testing driving flag, steering, and acceleration")
        print("Use 'ros2 run <package> <node>' for normal operation")
        test_mode()
    else:
        main()