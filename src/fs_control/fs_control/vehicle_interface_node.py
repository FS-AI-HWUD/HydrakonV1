import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

class VehicleInterfaceNode(Node):
    """
    Vehicle Interface for Formula Student - ADS-DV CAN Interface
    
    Converts control commands to AckermannDriveStamped messages for hydrakon_can:
    - Accepts acceleration commands (m/s²) from PID controller
    - Accepts steering commands (degrees) from planner
    - Publishes AckermannDriveStamped to hydrakon_can/command
    
    Subscribes to: 
    - /acceleration_cmd (Float64) - from PID controller
    - /planning/reference_steering (Float64) - from planner (in degrees)
    
    Publishes to: 
    - /hydrakon_can/command (AckermannDriveStamped) - to ADS-DV CAN interface
    """
    
    def __init__(self):
        super().__init__('vehicle_interface')
        
        # Parameters
        self.declare_parameter('control_frequency', 50.0)  # Hz
        
        # Get parameters
        self.control_freq = self.get_parameter('control_frequency').value
        
        # Subscribers
        self.acceleration_sub = self.create_subscription(
            Float64, '/acceleration_cmd', self.acceleration_callback, 10)
        self.steering_sub = self.create_subscription(
            Float64, '/planning/reference_steering', self.steering_callback, 10)
        
        # Publisher - ADS-DV CAN interface
        self.command_pub = self.create_publisher(
            AckermannDriveStamped, '/hydrakon_can/command', 10)
        
        # Command state
        self.current_acceleration = 0.0  # m/s²
        self.current_steering = 0.0      # radians
        
        # Synchronization - store latest timestamps
        self.last_acceleration_msg = None
        self.last_steering_msg = None
        
        # Control timer - publish commands to ADS-DV
        self.control_timer = self.create_timer(1.0/self.control_freq, self.publish_vehicle_command)
        
        self.get_logger().info("🏎️  Vehicle Interface Node - ADS-DV CAN Interface")
        self.get_logger().info(f"Control frequency: {self.control_freq} Hz")
    
    def acceleration_callback(self, msg):
        """Receive acceleration command from PID controller"""
        self.current_acceleration = msg.data  # m/s²
        self.last_acceleration_msg = self.get_clock().now()
        self.get_logger().debug(f"Acceleration command: {self.current_acceleration:+.2f} m/s²")
    
    def steering_callback(self, msg):
        """Receive steering command from planner (in degrees) and convert to radians"""
        steering_degrees = msg.data
        steering_radians = np.radians(steering_degrees)  # Convert degrees to radians
        self.current_steering = steering_radians
        self.last_steering_msg = self.get_clock().now()
        self.get_logger().debug(f"Steering command: {steering_degrees:+.1f}° -> {self.current_steering:+.3f} rad")
    
    def publish_vehicle_command(self):
        """Publish AckermannDriveStamped command to ADS-DV CAN interface"""
        try:
            # Create AckermannDriveStamped message
            cmd_msg = AckermannDriveStamped()
            
            # Use the most recent timestamp for synchronization
            current_time = self.get_clock().now()
            cmd_msg.header.stamp = current_time.to_msg()
            cmd_msg.header.frame_id = "base_link"
            
            # AckermannDrive fields
            cmd_msg.drive.steering_angle = float(self.current_steering)  # radians
            cmd_msg.drive.steering_angle_velocity = 0.0  # Not used
            cmd_msg.drive.speed = 0.0  # Not used - we use acceleration
            cmd_msg.drive.acceleration = float(self.current_acceleration)  # m/s²
            cmd_msg.drive.jerk = 0.0  # Not used
            
            # Publish command
            self.command_pub.publish(cmd_msg)
            
            # Debug logging
            if hasattr(self, 'log_counter'):
                self.log_counter += 1
            else:
                self.log_counter = 0
            
            if self.log_counter % 50 == 0:  # Every 50 cycles (1 second at 50Hz)
                self.get_logger().info(
                    f"🚗 Command -> Accel: {self.current_acceleration:+.2f} m/s² | "
                    f"Steer: {self.current_steering:+.3f} rad ({np.degrees(self.current_steering):+.1f}°)"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error publishing vehicle command: {e}")
    
    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info("Vehicle Interface Node shutting down")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VehicleInterfaceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Vehicle Interface shutting down...")
    except Exception as e:
        print(f"❌ Error in Vehicle Interface: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()