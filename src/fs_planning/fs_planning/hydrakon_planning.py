#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String, Bool
import re

class CombinedController(Node):
    def should_run_controller(self):
        """Check if controller should run based on AMI state"""
        if self.current_ami_state is None:
            return False
        
        # Run when AMI state is AUTOCROSS or TRACKDRIVE
        return self.current_ami_state in ['AUTOCROSS', 'TRACKDRIVE']
    
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
        
        self.mission_completed_publisher = self.create_publisher(
            Bool,
            '/hydrakon_can/is_mission_completed',
            10
        )
        
        # Camera parameters
        self.image_width = 1280
        self.image_height = 720
        self.image_center_x = self.image_width // 2
        
        # Steering parameters
        self.max_steering_angle = 0.4
        self.steering_gain = 0.003
        self.min_cone_distance = 50
        self.estimated_track_width_pixels = 240  # Standard Formula Student track width (~3.5m) in pixels
        
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
        
        # State tracking for AMI control
        self.current_as_state = None
        self.current_ami_state = None
        
        # Lap counting variables
        self.lap_count = 0
        self.max_laps_autocross = 2
        self.max_laps_trackdrive = 11
        self.last_orange_gate_time = 0.0
        self.cooldown_duration = 2.0  # 2 seconds cooldown between gate passages
        
        # NEW: Track orange cone positions for lap counting
        self.previous_orange_positions = []
        self.lap_detection_threshold = 0.7  # Threshold for detecting lap completion
        self.min_gate_approach_distance = self.image_height * 0.6  # Orange cones must be in bottom 40% of image
        
        # Mission completion variables
        self.mission_completed = False
        self.brake_timer = None
        self.brake_start_time = None
        self.brake_duration = 2.0  # 2 seconds
        
        # Last steering angle for continuity
        self.last_steering_angle = 0.0
        
        # Midpoint steering parameters
        self.midpoint_smoothing = 0.7  # Smoothing factor for midpoint calculation
        self.last_midpoint_x = None
        
        self.get_logger().info('Combined Controller initialized')
        self.get_logger().info('- Camera-based steering control')
        self.get_logger().info('- Runs when AMI state is AUTOCROSS or TRACKDRIVE')
        self.get_logger().info('- AUTOCROSS: 2 laps max, TRACKDRIVE: 11 laps max')
        self.get_logger().info('- Lap counting by passing through orange cones (movement-based detection)')
        self.get_logger().info('- Enhanced midpoint steering between yellow and blue cones')
        self.get_logger().info('- FIXED: Acceleration in curves - accelerates when ANY cones detected')
        self.get_logger().info('- Position validation: Yellow left, Blue right')
        self.get_logger().info('- Emergency brake when max laps completed')
        self.get_logger().info('- FIXED: Blue cones -> right turn, Yellow cones -> left turn')
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
            # Reset lap count when switching modes
            if self.current_ami_state != ami_state and ami_state in ['AUTOCROSS', 'TRACKDRIVE']:
                self.lap_count = 0
                self.mission_completed = False
                self.brake_start_time = None
                self.last_orange_gate_time = 0.0
                self.previous_orange_positions = []  # Reset tracking
                self.get_logger().info(f'Mode changed to {ami_state}, resetting lap count and mission status')
            
            self.current_as_state = as_state
            self.current_ami_state = ami_state
            self.get_logger().debug(f'Updated state - AS: {as_state}, AMI: {ami_state}')
    
    def find_closest_orange_gate(self, orange_cones):
        """Find the closest orange gate (pair of orange cones) or single cone"""
        if len(orange_cones) < 1:
            return None
        
        # If only one orange cone, treat it as a gate
        if len(orange_cones) == 1:
            cone = orange_cones[0]
            return {
                'midpoint_x': cone[0],
                'midpoint_y': cone[1],
                'distance': cone[1]  # Use y-coordinate as distance (closer to bottom = closer)
            }
        
        # Find pairs of orange cones that could form a gate
        best_gate = None
        max_distance = 0  # We want the closest gate (highest y-coordinate)
        
        for i in range(len(orange_cones)):
            for j in range(i + 1, len(orange_cones)):
                cone1 = orange_cones[i]
                cone2 = orange_cones[j]
                
                # Calculate midpoint between the two cones
                midpoint_x = (cone1[0] + cone2[0]) / 2.0
                midpoint_y = (cone1[1] + cone2[1]) / 2.0
                
                # Distance is based on y-coordinate (closer to bottom = closer)
                distance = midpoint_y
                
                # Check if this is a reasonable gate (cones not too far apart)
                cone_separation = abs(cone1[0] - cone2[0])
                if cone_separation < 300 and distance > max_distance:  # Reasonable gate width
                    best_gate = {
                        'midpoint_x': midpoint_x,
                        'midpoint_y': midpoint_y,
                        'distance': distance
                    }
                    max_distance = distance
        
        return best_gate
    
    def check_orange_gate_passage(self, orange_cones):
        """Check if vehicle has passed through orange gate using movement detection"""
        import time
        current_time = time.time()
        
        # Cooldown check to prevent multiple counts for same gate
        if current_time - self.last_orange_gate_time < self.cooldown_duration:
            return False
        
        # Check if we have orange cones
        if len(orange_cones) < 1:
            self.previous_orange_positions = []  # Reset if no cones
            return False
        
        # Find the closest orange gate
        best_gate = self.find_closest_orange_gate(orange_cones)
        if not best_gate:
            return False
        
        current_position = {
            'x': best_gate['midpoint_x'],
            'y': best_gate['midpoint_y'],
            'timestamp': current_time
        }
        
        # Check if gate is close enough to the car (in bottom portion of image)
        if best_gate['midpoint_y'] < self.min_gate_approach_distance:
            # Gate is too far away
            self.previous_orange_positions = []
            return False
        
        # Store current position for tracking
        self.previous_orange_positions.append(current_position)
        
        # Keep only recent positions (last 1 second)
        self.previous_orange_positions = [
            pos for pos in self.previous_orange_positions 
            if current_time - pos['timestamp'] < 1.0
        ]
        
        # Check if we have enough data points to detect passage
        if len(self.previous_orange_positions) < 3:
            return False
        
        # Check if orange gate has moved towards the car significantly
        # (y-coordinate should increase over time as gate approaches)
        positions = self.previous_orange_positions
        y_positions = [pos['y'] for pos in positions]
        
        # Calculate movement trend
        if len(y_positions) >= 3:
            early_y = sum(y_positions[:len(y_positions)//2]) / (len(y_positions)//2)
            recent_y = sum(y_positions[len(y_positions)//2:]) / (len(y_positions) - len(y_positions)//2)
            
            # Movement towards camera (increasing y) indicates passage
            movement = recent_y - early_y
            
            self.get_logger().debug(f"Orange gate movement: {movement:.2f} pixels (early: {early_y:.1f}, recent: {recent_y:.1f})")
            
            # If gate has moved significantly towards camera and is now close
            if movement > 50 and recent_y > self.image_height * 0.7:
                self.last_orange_gate_time = current_time
                self.previous_orange_positions = []  # Reset tracking
                self.get_logger().info(f"GATE PASSAGE DETECTED: Movement {movement:.1f} pixels, final position y={recent_y:.1f}")
                return True
        
        return False
    
    def check_lap_completion(self):
        """Check if maximum laps have been completed"""
        if self.current_ami_state == 'AUTOCROSS':
            return self.lap_count >= self.max_laps_autocross
        elif self.current_ami_state == 'TRACKDRIVE':
            return self.lap_count >= self.max_laps_trackdrive
        return False
    
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
        
        # Check for orange gate passage - LAP COUNTING OR FINISH
        if len(orange_cones) > 0:
            if self.check_orange_gate_passage(orange_cones):
                # Check if we've completed maximum laps
                if self.check_lap_completion():
                    if not self.mission_completed:
                        self.get_logger().warn(f'MAX LAPS COMPLETED! EMERGENCY BRAKE - {self.lap_count} laps done')
                        self.mission_completed = True
                        self.brake_start_time = self.get_clock().now()
                        self.publish_emergency_brake_command(msg.header.stamp)
                    return
                else:
                    # Count a lap
                    self.lap_count += 1
                    max_laps = self.max_laps_autocross if self.current_ami_state == 'AUTOCROSS' else self.max_laps_trackdrive
                    self.get_logger().info(f'LAP {self.lap_count} COMPLETED! Passed through orange gate ({self.lap_count}/{max_laps} laps in {self.current_ami_state} mode)')
        
        # Handle mission completion sequence
        if self.mission_completed and self.brake_start_time is not None:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.brake_start_time).nanoseconds / 1e9
            
            if elapsed_time >= self.brake_duration:
                # 2 seconds have passed, set mission completed flag
                self.publish_mission_completed()
                self.brake_start_time = None  # Prevent repeated publishing
                return
            else:
                # Still in brake period, continue braking
                self.publish_emergency_brake_command(msg.header.stamp)
                return
        
        # Calculate steering angle and acceleration
        steering_angle = self.calculate_steering_angle(yellow_cones, blue_cones)
        acceleration = self.calculate_acceleration(yellow_cones, blue_cones)
        
        # Only publish steering commands if AMI state is AUTOCROSS or TRACKDRIVE
        if self.should_run_controller():
            self.publish_steering_command(steering_angle, acceleration, msg.header.stamp)
            
            # Log detection info
            if len(yellow_cones) > 0 or len(blue_cones) > 0:
                accel_status = "ACCELERATING" if acceleration > 0 else "COASTING"
                max_laps = self.max_laps_autocross if self.current_ami_state == 'AUTOCROSS' else self.max_laps_trackdrive
                self.get_logger().info(f'{self.current_ami_state} ACTIVE (Lap {self.lap_count}/{max_laps}): {len(yellow_cones)} yellow, {len(blue_cones)} blue cones. Steering: {steering_angle:.3f} rad, {accel_status}: {acceleration:.1f}')
        else:
            # Send zero steering and acceleration when not in AUTOCROSS/TRACKDRIVE mode
            self.publish_steering_command(0.0, 0.0, msg.header.stamp)
            if len(yellow_cones) > 0 or len(blue_cones) > 0:
                self.get_logger().debug(f'NOT RACING: Cones detected but AMI state is {self.current_ami_state}, not running controller')
    
    def calculate_acceleration(self, yellow_cones, blue_cones):
        """Calculate acceleration based on cone detection - FIXED to accelerate in curves"""
        
        # FIXED: Accelerate when ANY cones are detected (not just pairs)
        # This allows acceleration through curves where only one cone type is visible
        
        # Case 1: Both yellow and blue cones detected - MAXIMUM ACCELERATION
        if len(yellow_cones) > 0 and len(blue_cones) > 0:
            # Find closest cones for position validation
            closest_yellow = self.find_closest_cone(yellow_cones)
            closest_blue = self.find_closest_cone(blue_cones)
            
            if closest_yellow is None or closest_blue is None:
                return self.cone_pair_acceleration * 0.8  # Still accelerate even if validation fails
            
            yellow_x = closest_yellow[0]
            blue_x = closest_blue[0]
            
            # More lenient position validation - allow for some overlap
            # Yellow should generally be on the left, blue on the right
            separation_threshold = 50  # Minimum separation in pixels
            
            if abs(yellow_x - blue_x) > separation_threshold:
                # Check if yellow is generally on the left side of blue
                if yellow_x < blue_x:
                    self.get_logger().debug(f'Valid cone pair: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f}, separation: {abs(yellow_x-blue_x):.1f} - MAX ACCELERATION')
                    return self.cone_pair_acceleration  # Full acceleration
                else:
                    self.get_logger().debug(f'Reversed cone positioning: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f} - Good acceleration')
                    return self.cone_pair_acceleration * 0.8  # Still good acceleration
            else:
                self.get_logger().debug(f'Cones close together: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f}, separation: {abs(yellow_x-blue_x):.1f} - Good acceleration')
                return self.cone_pair_acceleration * 0.8  # Good acceleration when cones are close
        
        # Case 2: Only yellow cones detected - CURVE ACCELERATION (left curve)
        elif len(yellow_cones) > 0:
            closest_yellow = self.find_closest_cone(yellow_cones)
            if closest_yellow is not None:
                self.get_logger().debug(f'Yellow cones only: Yellow at {closest_yellow[0]:.1f} - CURVE ACCELERATION (left curve)')
                return self.cone_pair_acceleration * 0.7  # Good acceleration through left curves
            
        # Case 3: Only blue cones detected - CURVE ACCELERATION (right curve)
        elif len(blue_cones) > 0:
            closest_blue = self.find_closest_cone(blue_cones)
            if closest_blue is not None:
                self.get_logger().debug(f'Blue cones only: Blue at {closest_blue[0]:.1f} - CURVE ACCELERATION (right curve)')
                return self.cone_pair_acceleration * 0.7  # Good acceleration through right curves
        
        # Case 4: No cones detected - COAST/MAINTAIN SPEED
        self.get_logger().debug('No cones detected - COASTING')
        return self.default_acceleration
    
    def calculate_steering_angle(self, yellow_cones, blue_cones):
        """Calculate steering angle with enhanced midpoint targeting - FIXED DIRECTIONS"""
        
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
        
        # Case 2: Only yellow cones detected - AVOID LEFT, turn RIGHT to get back on track
        elif len(yellow_cones) > 0:
            closest_yellow = self.find_closest_cone(yellow_cones)
            if closest_yellow is not None:
                yellow_x = closest_yellow[0]
                
                # FIXED: When seeing yellow cones (track boundary on left), 
                # aim to the RIGHT of the yellow cones to stay on track
                estimated_track_width_pixels = 240  # Estimated track width in pixels (3.5m real world)
                target_x = yellow_x + (estimated_track_width_pixels / 2)  # Aim RIGHT of yellow cones
                
                steering_mode = "avoid_yellow_left"
                self.get_logger().debug(f'Yellow avoidance: Yellow at {yellow_x:.1f}, Target (right of yellow): {target_x:.1f}')
        
        # Case 3: Only blue cones detected - AVOID RIGHT, turn LEFT to get back on track
        elif len(blue_cones) > 0:
            closest_blue = self.find_closest_cone(blue_cones)
            if closest_blue is not None:
                blue_x = closest_blue[0]
                
                # FIXED: When seeing blue cones (track boundary on right),
                # aim to the LEFT of the blue cones to stay on track
                estimated_track_width_pixels = 240  # Estimated track width in pixels (3.5m real world)
                target_x = blue_x - (estimated_track_width_pixels / 2)  # Aim LEFT of blue cones
                
                steering_mode = "avoid_blue_right"
                self.get_logger().debug(f'Blue avoidance: Blue at {blue_x:.1f}, Target (left of blue): {target_x:.1f}')
        
        # Convert target position to steering angle
        if target_x is not None:
            error_pixels = target_x - self.image_center_x
            # FIXED: Correct steering direction
            # Positive error (target right of center) -> positive steering (turn right)
            # Negative error (target left of center) -> negative steering (turn left)
            steering_angle = error_pixels * self.steering_gain  # Removed negative sign
            
            # Apply steering angle limits
            steering_angle = max(-self.max_steering_angle, 
                               min(self.max_steering_angle, steering_angle))
            
            # Store for continuity
            self.last_steering_angle = steering_angle
            
            # Log steering decision with direction
            direction = "RIGHT" if steering_angle > 0 else "LEFT" if steering_angle < 0 else "STRAIGHT"
            self.get_logger().debug(f'Steering: Mode={steering_mode}, Target={target_x:.1f}, Error={error_pixels:.1f}, Angle={steering_angle:.3f} ({direction})')
            
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
    
    def publish_mission_completed(self):
        """Publish mission completed flag"""
        msg = Bool()
        msg.data = True
        self.mission_completed_publisher.publish(msg)
        self.get_logger().info('MISSION COMPLETED FLAG SET TO TRUE - All laps finished!')
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
    
    mission_completed_publisher = node.create_publisher(
        Bool,
        '/hydrakon_can/is_mission_completed',
        10
    )
    
    node.get_logger().info('TEST MODE: Fixed Combined Controller Test')
    node.get_logger().info('FIXES: Blue cones -> right turn, Yellow cones -> left turn')
    node.get_logger().info('FIXES: Improved lap counting using orange cone movement detection')
    node.get_logger().info('Testing lap counting for AUTOCROSS (2 laps) and TRACKDRIVE (11 laps)')
    node.get_logger().info('Testing mission completion flag after 2 second brake')
    node.get_logger().info('Testing fixed steering directions and improved lap detection...')
    node.get_logger().info('NOTE: In normal operation, controller only runs when AMI state is AUTOCROSS or TRACKDRIVE')
    node.get_logger().info('Press Ctrl+C to stop')
    
    import time
    test_sequence = [
        {'steering': 0.0, 'acceleration': 0.0, 'description': 'Straight ahead'},
        {'steering': 0.1, 'acceleration': 0.9, 'description': 'Right turn with acceleration (blue cones detected)'},
        {'steering': 0.2, 'acceleration': 0.9, 'description': 'Sharp right with acceleration'},
        {'steering': 0.0, 'acceleration': 0.9, 'description': 'Straight with acceleration'},
        {'steering': -0.1, 'acceleration': 0.9, 'description': 'Left turn with acceleration (yellow cones detected)'},
        {'steering': -0.2, 'acceleration': 0.0, 'description': 'Sharp left, single cone'},
        {'steering': 0.0, 'acceleration': 0.0, 'brake': 60.0, 'description': 'Emergency brake test (max laps)'},
    ]
    
    sequence_index = 0
    
    try:
        while rclpy.ok():
            test_data = test_sequence[sequence_index % len(test_sequence)]
            
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
            
            node.get_logger().info(f'Published - {test_data["description"]}: Steering: {test_data["steering"]:.1f} rad, {status}')
            
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
        print("Running in TEST MODE - testing FIXED steering directions and improved lap counting")
        print("Use 'ros2 run <package> <node>' for normal operation")
        test_mode()
    else:
        main()