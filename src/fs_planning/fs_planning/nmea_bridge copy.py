#!/usr/bin/env python3
"""
CHCNAV INS NMEA Bridge (Improved)
Receives NMEA data via TCP on port 9906 and publishes to ROS2 topics
Uses standard GPGGA/GPRMC for GPS + custom GPCHC for IMU data
Part of Hydrakon Formula Student fs_planning package
"""

import socket
import threading
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from geometry_msgs.msg import TwistWithCovarianceStamped, Vector3, Quaternion, PoseStamped, QuaternionStamped
from std_msgs.msg import Header, Float64


class CHCNAVINSBridge(Node):
    def __init__(self):
        super().__init__('chcnav_ins_bridge')
        
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers - matching your launch file topic whitelist
        self.gnss_pub = self.create_publisher(NavSatFix, '/ins/gnss', sensor_qos)
        self.nav_pub = self.create_publisher(TwistWithCovarianceStamped, '/ins/nav', sensor_qos)
        self.heading_pub = self.create_publisher(QuaternionStamped, '/ins/heading', sensor_qos)
        self.velocity_pub = self.create_publisher(TwistWithCovarianceStamped, '/ins/velocity', sensor_qos)
        self.imu_pub = self.create_publisher(Imu, '/ins/imu', sensor_qos)
        
        # TCP Server setup for CHCNAV CGI-410
        self.tcp_host = '0.0.0.0'  # Listen on all interfaces
        self.tcp_port = 9906  # Your working port
        self.server_socket = None
        self.running = False
        
        # Data storage for INS system
        self.latest_gnss_data = {}
        self.latest_velocity_data = {}
        self.latest_heading_data = {}
        self.message_counts = {'GGA': 0, 'RMC': 0, 'CHC': 0}
        
        # Start TCP server
        self.start_tcp_server()
        
        self.get_logger().info('CHCNAV INS Bridge initialized - listening on port 9906')
        self.get_logger().info('Using: GPGGA/GPRMC for GPS + GPCHC for IMU data')
        self.get_logger().info('Publishing to: /ins/gnss, /ins/nav, /ins/heading, /ins/velocity, /ins/imu')

    def start_tcp_server(self):
        """Start the TCP server in a separate thread"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.tcp_host, self.tcp_port))
            self.server_socket.listen(5)
            self.running = True
            
            self.get_logger().info(f'CHCNAV INS TCP Server listening on {self.tcp_host}:{self.tcp_port}')
            
            # Start server thread
            server_thread = threading.Thread(target=self.server_loop, daemon=True)
            server_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'Failed to start TCP server: {str(e)}')

    def server_loop(self):
        """Main server loop to accept connections"""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                self.get_logger().info(f'CHCNAV connected from {client_address}')
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    self.get_logger().error(f'Server error: {str(e)}')

    def handle_client(self, client_socket, client_address):
        """Handle individual client connections"""
        buffer = ""
        
        try:
            while self.running:
                data = client_socket.recv(1024).decode('utf-8', errors='ignore')
                if not data:
                    break
                
                buffer += data
                
                # Process complete NMEA sentences
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line.startswith('$'):
                        self.parse_nmea_sentence(line)
                        
                # Log message statistics every 100 messages
                total_messages = sum(self.message_counts.values())
                if total_messages % 100 == 0 and total_messages > 0:
                    self.get_logger().info(f'Messages - GGA: {self.message_counts["GGA"]}, '
                                         f'RMC: {self.message_counts["RMC"]}, '
                                         f'CHC: {self.message_counts["CHC"]}')
                        
        except Exception as e:
            self.get_logger().error(f'Client handling error: {str(e)}')
        finally:
            client_socket.close()
            self.get_logger().info(f'CHCNAV {client_address} disconnected')

    def parse_nmea_sentence(self, sentence):
        """Parse NMEA sentences and extract data"""
        try:
            # Verify checksum if present
            if '*' in sentence:
                msg_part, checksum = sentence.rsplit('*', 1)
                if not self.verify_checksum(msg_part, checksum):
                    return
            else:
                msg_part = sentence
            
            parts = msg_part.split(',')
            msg_type = parts[0]
            
            if msg_type in ['$GPGGA', '$GNGGA']:
                self.parse_gpgga(parts)
                self.message_counts['GGA'] += 1
            elif msg_type in ['$GPRMC', '$GNRMC']:
                self.parse_gprmc(parts)
                self.message_counts['RMC'] += 1
            elif msg_type == '$GPCHC':
                self.parse_gpchc(parts)
                self.message_counts['CHC'] += 1
                
        except Exception as e:
            self.get_logger().error(f'Error parsing NMEA sentence {sentence}: {str(e)}')

    def verify_checksum(self, msg_part, checksum_hex):
        """Verify NMEA checksum"""
        try:
            calc_checksum = 0
            for char in msg_part[1:]:  # Skip the '$'
                calc_checksum ^= ord(char)
            
            expected_checksum = int(checksum_hex, 16)
            return calc_checksum == expected_checksum
        except:
            return False

    def parse_gpgga(self, parts):
        """Parse standard GPGGA sentence for GPS position data"""
        try:
            if len(parts) < 14:
                return
            
            lat_raw = parts[2]
            lat_dir = parts[3]
            lon_raw = parts[4]
            lon_dir = parts[5]
            fix_quality = int(parts[6]) if parts[6] else 0
            num_sats = int(parts[7]) if parts[7] else 0
            hdop = float(parts[8]) if parts[8] else 0.0
            altitude = float(parts[9]) if parts[9] else 0.0
            
            # Convert coordinates using standard NMEA format
            if lat_raw and lon_raw:
                latitude = self.convert_nmea_coord(lat_raw, lat_dir, coord_type='lat')
                longitude = self.convert_nmea_coord(lon_raw, lon_dir, coord_type='lon')
                
                self.latest_gnss_data.update({
                    'latitude': latitude,
                    'longitude': longitude,
                    'altitude': altitude,
                    'fix_quality': fix_quality,
                    'num_satellites': num_sats,
                    'hdop': hdop
                })
                
                self.publish_gnss()
                
                # Log occasionally
                if self.message_counts['GGA'] % 50 == 1:
                    quality_str = {
                        0: 'Invalid', 1: 'GPS Fix', 2: 'DGPS Fix', 
                        4: 'RTK Fixed', 5: 'RTK Float'
                    }.get(fix_quality, f'Unknown({fix_quality})')
                    
                    self.get_logger().info(
                        f'GNSS: {latitude:.6f}, {longitude:.6f}, {altitude:.1f}m | '
                        f'Quality: {quality_str} | Sats: {num_sats}'
                    )
                
        except Exception as e:
            self.get_logger().error(f'Error parsing GPGGA: {str(e)}')

    def parse_gprmc(self, parts):
        """Parse standard GPRMC sentence for velocity and course data"""
        try:
            if len(parts) < 10:
                return
            
            status = parts[2]
            speed_knots = parts[7]
            course = parts[8]
            
            if status == 'A' and speed_knots:  # Active/Valid
                # Convert speed from knots to m/s
                speed_ms = float(speed_knots) * 0.514444
                course_deg = float(course) if course else 0.0
                course_rad = math.radians(course_deg)
                
                # Calculate velocity components (North-East frame)
                vel_north = speed_ms * math.cos(course_rad)
                vel_east = speed_ms * math.sin(course_rad)
                
                self.latest_velocity_data.update({
                    'speed': speed_ms,
                    'course': course_deg,
                    'vel_north': vel_north,
                    'vel_east': vel_east
                })
                
                self.publish_nav_velocity()
                
                # Log occasionally
                if self.message_counts['RMC'] % 50 == 1:
                    self.get_logger().info(
                        f'NAV: Speed: {speed_ms:.2f} m/s, Course: {course_deg}°, '
                        f'Vel: N={vel_north:.2f}, E={vel_east:.2f}'
                    )
                
        except Exception as e:
            self.get_logger().error(f'Error parsing GPRMC: {str(e)}')

    def parse_gpchc(self, parts):
        """Parse custom GPCHC sentence for IMU data"""
        try:
            if len(parts) < 20:
                return
            
            # Based on CHCNAV CGI-410 data structure
            gyro_x = float(parts[3]) if parts[3] else 0.0
            gyro_y = float(parts[4]) if parts[4] else 0.0  
            gyro_z = float(parts[5]) if parts[5] else 0.0
            accel_x = float(parts[6]) if parts[6] else 0.0
            accel_y = float(parts[7]) if parts[7] else 0.0
            accel_z = float(parts[8]) if parts[8] else 0.0
            roll = float(parts[9]) if parts[9] else 0.0
            pitch = float(parts[10]) if parts[10] else 0.0
            yaw = float(parts[11]) if parts[11] else 0.0
            
            # Store IMU data
            imu_data = {
                'angular_velocity': [gyro_x, gyro_y, gyro_z],
                'linear_acceleration': [accel_x, accel_y, accel_z],
                'orientation_euler': [roll, pitch, yaw]
            }
            
            # Store heading data  
            self.latest_heading_data.update({
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw
            })
            
            self.publish_imu(imu_data)
            self.publish_heading()
            
            # Log occasionally
            if self.message_counts['CHC'] % 50 == 1:
                self.get_logger().info(f'IMU: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°')
            
        except Exception as e:
            self.get_logger().error(f'Error parsing GPCHC: {str(e)}')

    def convert_nmea_coord(self, coord_str, direction, coord_type):
        """Convert NMEA coordinate format to decimal degrees"""
        try:
            if len(coord_str) < 4:
                return 0.0
            
            # Standard NMEA format
            if coord_type == 'lat':  # Latitude: DDMM.MMMM
                degrees = int(coord_str[:2])
                minutes = float(coord_str[2:])
            else:  # Longitude: DDDMM.MMMM  
                degrees = int(coord_str[:3])
                minutes = float(coord_str[3:])
            
            decimal_degrees = degrees + minutes / 60.0
            
            if direction in ['S', 'W']:
                decimal_degrees = -decimal_degrees
                
            return decimal_degrees
        except Exception as e:
            self.get_logger().error(f'Coordinate conversion error for {coord_str}: {str(e)}')
            return 0.0

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        # Convert degrees to radians
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return Quaternion(x=x, y=y, z=z, w=w)

    def publish_gnss(self):
        """Publish GNSS data to ROS2 topic"""
        if not self.latest_gnss_data:
            return
        
        msg = NavSatFix()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'gps'
        
        msg.latitude = self.latest_gnss_data.get('latitude', 0.0)
        msg.longitude = self.latest_gnss_data.get('longitude', 0.0)
        msg.altitude = self.latest_gnss_data.get('altitude', 0.0)
        
        # Status based on fix quality
        fix_quality = self.latest_gnss_data.get('fix_quality', 0)
        if fix_quality == 0:
            msg.status.status = -1  # STATUS_NO_FIX
        elif fix_quality == 1:
            msg.status.status = 0   # STATUS_FIX
        elif fix_quality == 2:
            msg.status.status = 2   # STATUS_SBAS_FIX
        elif fix_quality in [4, 5]:
            msg.status.status = 1   # STATUS_GBAS_FIX (RTK)
        
        msg.status.service = 1  # SERVICE_GPS
        
        # Set covariance based on fix quality
        fix_quality = self.latest_gnss_data.get('fix_quality', 0)
        if fix_quality in [4, 5]:  # RTK
            position_covariance = 0.01  # 1cm accuracy
        elif fix_quality == 2:  # DGPS
            position_covariance = 1.0   # 1m accuracy
        else:  # Standard GPS
            position_covariance = 9.0   # 3m accuracy
            
        msg.position_covariance = [
            float(position_covariance), 0.0, 0.0,
            0.0, float(position_covariance), 0.0,
            0.0, 0.0, float(position_covariance * 4.0)
        ]
        msg.position_covariance_type = 2  # COVARIANCE_TYPE_DIAGONAL_KNOWN
        
        self.gnss_pub.publish(msg)

    def publish_nav_velocity(self):
        """Publish navigation velocity to /ins/nav"""
        if not self.latest_velocity_data:
            return
        
        msg = TwistWithCovarianceStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'gps'
        
        # Use North-East velocity components
        msg.twist.twist.linear.x = self.latest_velocity_data.get('vel_north', 0.0)
        msg.twist.twist.linear.y = self.latest_velocity_data.get('vel_east', 0.0)
        msg.twist.twist.linear.z = 0.0
        
        # Set covariance
        vel_cov = 0.1  # 0.1 m/s accuracy
        msg.twist.covariance = [
            float(vel_cov), 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, float(vel_cov), 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, float(vel_cov), 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.1, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.1
        ]
        
        self.nav_pub.publish(msg)

    def publish_heading(self):
        """Publish heading data to /ins/heading"""
        if not self.latest_heading_data:
            return
        
        msg = QuaternionStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'gps'
        
        # Convert Euler angles to quaternion
        roll = self.latest_heading_data.get('roll', 0.0)
        pitch = self.latest_heading_data.get('pitch', 0.0)
        yaw = self.latest_heading_data.get('yaw', 0.0)
        
        msg.quaternion = self.euler_to_quaternion(roll, pitch, yaw)
        
        self.heading_pub.publish(msg)

    def publish_imu(self, imu_data):
        """Publish IMU data to /ins/imu"""
        msg = Imu()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        
        # Angular velocity (rad/s)
        angular_vel = imu_data.get('angular_velocity', [0, 0, 0])
        msg.angular_velocity = Vector3(x=angular_vel[0], y=angular_vel[1], z=angular_vel[2])
        
        # Linear acceleration (m/s²)
        linear_accel = imu_data.get('linear_acceleration', [0, 0, 0])
        msg.linear_acceleration = Vector3(x=linear_accel[0], y=linear_accel[1], z=linear_accel[2])
        
        # Orientation from Euler angles
        euler = imu_data.get('orientation_euler', [0, 0, 0])
        msg.orientation = self.euler_to_quaternion(euler[0], euler[1], euler[2])
        
        # Set covariance matrices (diagonal) - using float values
        msg.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        
        self.imu_pub.publish(msg)

    def destroy_node(self):
        """Clean shutdown"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = CHCNAVINSBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()