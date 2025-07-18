#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String, Bool
import math
import numpy as np
import re
from collections import defaultdict

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
        self.ORANGE_CONE = 2
        self.LARGE_ORANGE_CONE = 3
        
        # Default drive parameters
        self.default_speed = 0.0
        self.default_acceleration = 0.0
        self.emergency_brake_value = 60.0  # Brake value for emergency stop
        self.cone_pair_acceleration = 0.9  # Acceleration when both cone types are detected
        
        # State tracking for driving flag
        self.current_as_state = None
        self.current_ami_state = None
        
        # Last steering angle for continuity
        self.last_steering_angle = 0.0
        
        # Midpoint steering parameters
        self.midpoint_smoothing = 0.7  # Smoothing factor for midpoint calculation
        self.last_midpoint_x = None
        
        self.get_logger().info('Combined Controller initialized')
        self.get_logger().info('- Camera-based steering control')
        self.get_logger().info('- AMI state monitoring with 20Hz driving flag')
        self.get_logger().info('- Enhanced midpoint steering between yellow and blue cones')
        self.get_logger().info('- Acceleration when cone pairs detected')
        self.get_logger().info('- Position validation: Yellow left, Blue right')
        self.get_logger().info('- Emergency brake on orange cone detection')
        self.get_logger().info(f'Emergency brake value: {self.emergency_brake_value}')
        self.get_logger().info(f'Image dimensions: {self.image_width}x{self.image_height}')
        self.get_logger().info(f'Steering gain: {self.steering_gain}, Max angle: {self.max_steering_angle}')
        self.get_logger().info(f'Cone pair acceleration: {self.cone_pair_acceleration}')
        self.get_logger().info(f'Midpoint smoothing factor: {self.midpoint_smoothing}')
    
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
        orange_cones = []
        
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
                    elif class_id == self.ORANGE_CONE or class_id == self.LARGE_ORANGE_CONE:
                        orange_cones.append((center_x, center_y, confidence))
        
        # Check for orange cones (finish line) - EMERGENCY BRAKE
        if len(orange_cones) > 0:
            self.get_logger().warn(f'ORANGE CONES DETECTED! EMERGENCY BRAKE - {len(orange_cones)} orange cones found')
            self.publish_emergency_brake_command(msg.header.stamp)
            return
        
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
        """Calculate acceleration based on cone detection with enhanced position validation"""
        # Only accelerate when both yellow and blue cones are detected
        if len(yellow_cones) > 0 and len(blue_cones) > 0:
            # Find closest cones for position validation
            closest_yellow = self.find_closest_cone(yellow_cones)
            closest_blue = self.find_closest_cone(blue_cones)
            
            if closest_yellow is None or closest_blue is None:
                return self.default_acceleration
            
            yellow_x = closest_yellow[0]
            blue_x = closest_blue[0]
            
            # More lenient position validation - allow for some overlap
            # Yellow should generally be on the left, blue on the right
            separation_threshold = 50  # Minimum separation in pixels
            
            if abs(yellow_x - blue_x) > separation_threshold:
                # Check if yellow is generally on the left side of blue
                if yellow_x < blue_x:
                    self.get_logger().debug(f'Valid cone pair: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f}, separation: {abs(yellow_x-blue_x):.1f} - ACCELERATING')
                    return self.cone_pair_acceleration
                else:
                    self.get_logger().debug(f'Reversed cone positioning: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f} - Slow acceleration')
                    return self.cone_pair_acceleration * 0.5  # Reduced acceleration for unclear positioning
            else:
                self.get_logger().debug(f'Cones too close: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f}, separation: {abs(yellow_x-blue_x):.1f} - Normal acceleration')
                return self.cone_pair_acceleration * 0.7  # Moderate acceleration when cones are close
        else:
            return self.default_acceleration
    
    def calculate_steering_angle(self, yellow_cones, blue_cones):
        """Calculate steering angle with enhanced midpoint targeting"""
        
        # If no cones detected, return last steering angle for continuity
        if len(yellow_cones) == 0 and len(blue_cones) == 0:
            return self.last_steering_angle * 0.9  # Gradually reduce steering
        
        target_x = None
        steering_mode = "unknown"
        
        # Case 1: Both yellow and blue cones detected - MIDPOINT STEERING
        if len(yellow_cones) > 0 and len(blue_cones) > 0:
            # Find the best pair of cones for midpoint calculation
            closest_yellow = self.find_closest_cone(yellow_cones)
            closest_blue = self.find_closest_cone(blue_cones)
            
            if closest_yellow is not None and closest_blue is not None:
                yellow_x = closest_yellow[0]
                blue_x = closest_blue[0]
                
                # Calculate raw midpoint
                raw_midpoint = (yellow_x + blue_x) / 2.0
                
                # Apply smoothing if we have a previous midpoint
                if self.last_midpoint_x is not None:
                    target_x = (self.midpoint_smoothing * self.last_midpoint_x + 
                               (1 - self.midpoint_smoothing) * raw_midpoint)
                else:
                    target_x = raw_midpoint
                
                # Store for next iteration
                self.last_midpoint_x = target_x
                
                steering_mode = "midpoint"
                
                # Enhanced logging for midpoint steering
                self.get_logger().debug(f'Midpoint steering: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f}')
                self.get_logger().debug(f'Raw midpoint: {raw_midpoint:.1f}, Smoothed target: {target_x:.1f}')
        
        # Case 2: Only yellow cones detected
        elif len(yellow_cones) > 0:
            closest_yellow = self.find_closest_cone(yellow_cones)
            if closest_yellow is not None:
                yellow_x = closest_yellow[0]
                
                # Target point is to the right of yellow cone (assuming yellow is left boundary)
                offset = 120  # Increased offset for better track following
                target_x = yellow_x + offset
                
                steering_mode = "yellow_only"
                self.get_logger().debug(f'Yellow-only steering: Yellow at {yellow_x:.1f}, Target: {target_x:.1f}')
        
        # Case 3: Only blue cones detected
        elif len(blue_cones) > 0:
            closest_blue = self.find_closest_cone(blue_cones)
            if closest_blue is not None:
                blue_x = closest_blue[0]
                
                # Target point is to the left of blue cone (assuming blue is right boundary)
                offset = 120  # Increased offset for better track following
                target_x = blue_x - offset
                
                steering_mode = "blue_only"
                self.get_logger().debug(f'Blue-only steering: Blue at {blue_x:.1f}, Target: {target_x:.1f}')
        
        # Convert target position to steering angle
        if target_x is not None:
            error_pixels = target_x - self.image_center_x
            steering_angle = -error_pixels * self.steering_gain
            
            # Apply steering angle limits
            steering_angle = max(-self.max_steering_angle, 
                               min(self.max_steering_angle, steering_angle))
            
            # Store for continuity
            self.last_steering_angle = steering_angle
            
            # Log steering decision
            self.get_logger().debug(f'Steering calculation: Mode={steering_mode}, Target={target_x:.1f}, Error={error_pixels:.1f}, Angle={steering_angle:.3f}')
            
            return steering_angle
        
        # Fallback: gradually reduce last steering angle
        fallback_angle = self.last_steering_angle * 0.9
        self.get_logger().debug(f'Fallback steering: {fallback_angle:.3f}')
        return fallback_angle
    
    def find_closest_cone(self, cones):
        """Find the closest cone based on y-coordinate (bottom of image is closer)"""
        if not cones:
            return None
        
        # Closest cone is the one with the highest y-coordinate (bottom of image)
        closest_cone = max(cones, key=lambda cone: cone[1])
        return closest_cone
    
    def publish_emergency_brake_command(self, timestamp):
        """Publish emergency brake command when orange cones detected"""
        msg = AckermannDriveStamped()
        
        msg.header.stamp = timestamp
        msg.header.frame_id = 'base_link'
        
        # Emergency brake: 0 steering, 0 speed, 0 acceleration, 60.0 brake
        msg.drive.steering_angle = 0.0
        msg.drive.steering_angle_velocity = 0.0
        msg.drive.speed = 0.0
        msg.drive.acceleration = 0.0
        msg.drive.jerk = self.emergency_brake_value  # Using jerk field for brake value
        
        self.command_publisher.publish(msg)
        self.get_logger().warn(f'EMERGENCY BRAKE APPLIED: {self.emergency_brake_value}')
    
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
    
    node.get_logger().info('TEST MODE: Combined Controller Test with Enhanced Midpoint Steering')
    node.get_logger().info('Testing driving flag, midpoint steering, acceleration, and emergency brake...')
    node.get_logger().info('Press Ctrl+C to stop')
    
    import time
    test_sequence = [
        {'driving': True, 'steering': 0.0, 'acceleration': 0.0, 'description': 'Straight ahead'},
        {'driving': True, 'steering': 0.1, 'acceleration': 0.9, 'description': 'Right turn with acceleration'},
        {'driving': True, 'steering': 0.2, 'acceleration': 0.9, 'description': 'Sharp right with acceleration'},
        {'driving': True, 'steering': 0.0, 'acceleration': 0.9, 'description': 'Straight with acceleration'},
        {'driving': True, 'steering': -0.1, 'acceleration': 0.9, 'description': 'Left turn with acceleration'},
        {'driving': True, 'steering': -0.2, 'acceleration': 0.0, 'description': 'Sharp left, single cone'},
        {'driving': False, 'steering': 0.0, 'acceleration': 0.0, 'description': 'Not driving'},
        {'driving': True, 'steering': 0.0, 'acceleration': 0.0, 'brake': 60.0, 'description': 'Emergency brake test'},
    ]
    
    sequence_index = 0
    
    try:
        while rclpy.ok():
            test_data = test_sequence[sequence_index % len(test_sequence)]
            
            # Publish driving flag
            flag_msg = Bool()
            flag_msg.data = test_data['driving']
            driving_flag_publisher.publish(flag_msg)
            
            # Publish steering command with acceleration or brake
            cmd_msg = AckermannDriveStamped()
            cmd_msg.header.stamp = node.get_clock().now().to_msg()
            cmd_msg.header.frame_id = 'base_link'
            cmd_msg.drive.steering_angle = float(test_data['steering'])
            cmd_msg.drive.steering_angle_velocity = 0.0
            cmd_msg.drive.speed = 0.0
            cmd_msg.drive.acceleration = float(test_data['acceleration'])
            
            # Check if this is an emergency brake test
            if 'brake' in test_data:
                cmd_msg.drive.jerk = float(test_data['brake'])
                status = f"EMERGENCY BRAKE: {test_data['brake']}"
            else:
                cmd_msg.drive.jerk = 0.0
                status = "ACCELERATING" if test_data['acceleration'] > 0 else "COASTING"
            
            command_publisher.publish(cmd_msg)
            
            node.get_logger().info(f'Published - {test_data["description"]}: Driving: {test_data["driving"]}, Steering: {test_data["steering"]:.1f} rad, {status}')
            
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
        print("Running in TEST MODE - testing driving flag, steering, acceleration, and emergency brake")
        print("Use 'ros2 run <package> <node>' for normal operation")
        test_mode()
    else:
        main()