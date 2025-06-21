#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from geometry_msgs.msg import TwistWithCovarianceStamped, QuaternionStamped
import socket
import math

class NMEABridge(Node):
    def __init__(self):
        super().__init__('nmea_bridge')
        
        # Create publishers for different data streams
        self.fix_publisher = self.create_publisher(NavSatFix, '/ins/gnss', 10)
        self.nav_publisher = self.create_publisher(TwistWithCovarianceStamped, '/ins/nav', 10)
        self.heading_publisher = self.create_publisher(QuaternionStamped, '/ins/heading', 10)
        
        # Setup TCP server (not client)
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', 9901))
            self.server_socket.listen(1)
            self.get_logger().info('TCP server listening on port 9901, waiting for CHCNAV...')
            self.get_logger().info('Publishing to: /ins/gnss, /ins/nav, /ins/heading')
            
            # Accept connection from CHCNAV
            self.sock, addr = self.server_socket.accept()
            self.sock.settimeout(1.0)
            self.get_logger().info(f'CHCNAV connected from {addr}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to create TCP server: {e}')
            return
        
        # Buffer for incomplete messages
        self.buffer = ""
        
        # Message counters for logging
        self.message_counts = {'GGA': 0, 'RMC': 0, 'HDT': 0}
        
        # Timer to read data
        self.timer = self.create_timer(0.1, self.read_nmea)
        
    def read_nmea(self):
        try:
            data = self.sock.recv(1024).decode('utf-8')
            self.buffer += data
            
            # Process complete lines
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                line = line.strip()
                
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    self.parse_gga(line)
                    self.message_counts['GGA'] += 1
                elif line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                    self.parse_rmc(line)
                    self.message_counts['RMC'] += 1
                elif line.startswith('$GPHDT') or line.startswith('$GNHDT'):
                    self.parse_hdt(line)
                    self.message_counts['HDT'] += 1
                    
            # Log message statistics every 30 messages
            total_messages = sum(self.message_counts.values())
            if total_messages % 30 == 0 and total_messages > 0:
                self.get_logger().info(f'Messages - GGA: {self.message_counts["GGA"]}, '
                                     f'RMC: {self.message_counts["RMC"]}, '
                                     f'HDT: {self.message_counts["HDT"]}')
                    
        except socket.timeout:
            # Normal timeout, continue
            pass
        except Exception as e:
            self.get_logger().error(f'Error reading NMEA: {e}')
    
    def parse_gga(self, sentence):
        """Parse GGA sentence for position data -> /ins/gnss"""
        try:
            parts = sentence.split(',')
            if len(parts) < 14:
                return
                
            # Extract fields
            lat_raw = parts[2]
            lat_dir = parts[3]
            lon_raw = parts[4]
            lon_dir = parts[5]
            quality = parts[6]
            num_sats = parts[7]
            hdop = parts[8]
            alt_raw = parts[9]
            
            if not lat_raw or not lon_raw:
                return
                
            # Convert DDMM.MMMM to decimal degrees
            lat_deg = int(lat_raw[:2])
            lat_min = float(lat_raw[2:])
            lat = lat_deg + lat_min / 60.0
            if lat_dir == 'S':
                lat = -lat
                
            lon_deg = int(lon_raw[:3])
            lon_min = float(lon_raw[3:])
            lon = lon_deg + lon_min / 60.0
            if lon_dir == 'W':
                lon = -lon
            
            alt = float(alt_raw) if alt_raw else 0.0
            
            # Create NavSatFix message
            msg = NavSatFix()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'gps'
            
            msg.latitude = lat
            msg.longitude = lon
            msg.altitude = alt
            
            # Set status based on GPS quality
            msg.status.status = NavSatStatus.STATUS_NO_FIX
            if quality == '1':
                msg.status.status = NavSatStatus.STATUS_FIX
            elif quality == '2':
                msg.status.status = NavSatStatus.STATUS_SBAS_FIX
            elif quality in ['4', '5']:
                msg.status.status = NavSatStatus.STATUS_GBAS_FIX  # RTK
            
            msg.status.service = NavSatStatus.SERVICE_GPS
            
            # Set covariance (rough estimates)
            if quality in ['4', '5']:  # RTK
                pos_cov = 0.01  # 1cm accuracy
            elif quality == '2':  # DGPS
                pos_cov = 1.0   # 1m accuracy
            else:  # Standard GPS
                pos_cov = 9.0   # 3m accuracy
                
            msg.position_covariance = [
                float(pos_cov), 0.0, 0.0,
                0.0, float(pos_cov), 0.0,
                0.0, 0.0, float(pos_cov * 4)  # Altitude less accurate
            ]
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            
            self.fix_publisher.publish(msg)
            
            # Log quality info occasionally
            if self.message_counts['GGA'] % 10 == 1:
                quality_str = {
                    '0': 'Invalid',
                    '1': 'GPS Fix',
                    '2': 'DGPS Fix', 
                    '4': 'RTK Fixed',
                    '5': 'RTK Float'
                }.get(quality, f'Unknown({quality})')
                
                self.get_logger().info(
                    f'GNSS: {lat:.6f}, {lon:.6f}, {alt:.1f}m | '
                    f'Quality: {quality_str} | Sats: {num_sats}'
                )
            
        except Exception as e:
            self.get_logger().error(f'Error parsing GGA: {e}')
    
    def parse_rmc(self, sentence):
        """Parse RMC sentence for velocity data -> /ins/nav"""
        try:
            parts = sentence.split(',')
            if len(parts) < 12:
                return
                
            status = parts[2]
            speed_knots = parts[7]
            course = parts[8]
            
            if status != 'A' or not speed_knots:  # A = Active/Valid
                return
                
            # Convert knots to m/s
            speed_ms = float(speed_knots) * 0.514444
            course_rad = math.radians(float(course)) if course else 0.0
            
            # Calculate velocity components (North-East frame)
            vel_north = speed_ms * math.cos(course_rad)  # North component
            vel_east = speed_ms * math.sin(course_rad)   # East component
            
            # Create velocity message
            msg = TwistWithCovarianceStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'gps'
            
            msg.twist.twist.linear.x = vel_north
            msg.twist.twist.linear.y = vel_east
            msg.twist.twist.linear.z = 0.0
            
            # Set covariance (rough estimates)
            vel_cov = 0.1  # 0.1 m/s accuracy
            msg.twist.covariance = [
                float(vel_cov), 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, float(vel_cov), 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, float(vel_cov), 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1
            ]
            
            self.nav_publisher.publish(msg)
            
            # Log velocity info occasionally
            if self.message_counts['RMC'] % 10 == 1:
                self.get_logger().info(
                    f'NAV: Speed: {speed_ms:.2f} m/s, Course: {course}°, '
                    f'Vel: N={vel_north:.2f}, E={vel_east:.2f}'
                )
            
        except Exception as e:
            self.get_logger().error(f'Error parsing RMC: {e}')
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion (without tf_transformations dependency)"""
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
        
        return [x, y, z, w]
    
    def parse_hdt(self, sentence):
        """Parse HDT sentence for heading data -> /ins/heading"""
        try:
            parts = sentence.split(',')
            if len(parts) < 3:
                return
                
            heading_deg_str = parts[1]
            true_indicator = parts[2].split('*')[0]  # Remove checksum
            
            if not heading_deg_str or true_indicator != 'T':
                return
                
            heading_deg = float(heading_deg_str)
            
            # Convert heading to radians and create quaternion
            heading_rad = math.radians(heading_deg)
            
            # Create quaternion from yaw angle (roll=0, pitch=0, yaw=heading)
            q = self.euler_to_quaternion(0, 0, heading_rad)
            
            # Create QuaternionStamped message
            msg = QuaternionStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'gps'
            
            msg.quaternion.x = q[0]
            msg.quaternion.y = q[1]
            msg.quaternion.z = q[2]
            msg.quaternion.w = q[3]
            
            self.heading_publisher.publish(msg)
            
            # Log heading info occasionally
            if self.message_counts['HDT'] % 10 == 1:
                self.get_logger().info(f'HEADING: {heading_deg:.1f}° true')
            
        except Exception as e:
            self.get_logger().error(f'Error parsing HDT: {e}')
    
    def __del__(self):
        if hasattr(self, 'sock'):
            self.sock.close()
        if hasattr(self, 'server_socket'):
            self.server_socket.close()

def main(args=None):
    rclpy.init(args=args)
    bridge = NMEABridge()
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        bridge.get_logger().info('Shutting down NMEA bridge...')
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
