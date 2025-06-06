#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os
import traceback
from ament_index_python.packages import get_package_share_directory

class LidarClusterNode(Node):
    def __init__(self):
        super().__init__('rslidar_cluster')
        
        # Topic names
        self.raw_topic = "/lidar/points"
        self.cluster_topic = "/perception/lidar_cluster" 
        self.cone_markers_topic = "/perception/cone_markers"
        
        # REFINED parameters for better cone detection
        self.eps = 0.4
        self.min_points = 3
        self.debug_mode = True
        
        # Distance filtering - reasonable range
        self.min_distance = 1.5
        self.max_distance = 12.0
        
        # FINE-TUNED cone validation for real LiDAR clustering behavior
        self.min_height = 0.10
        self.max_height = 0.50
        self.max_width = 0.45
        self.cone_min_points = 3
        self.cone_max_points = 100
        
        # Enhanced filtering parameters
        self.intensity_threshold = 8.0
        self.voxel_size = 0.12
        self.process_every_n = 1
        self.max_points_to_process = 1500
        
        # Advanced cone validation 
        self.min_density = 10.0
        self.max_density = 600.0
        self.max_aspect_ratio = 4.0
        self.min_volume = 0.0005
        
        # Real-time settings
        self.marker_lifetime_ns = 300000000  # 0.3 seconds
        self.enable_cluster_publishing = True   # Re-enable for debugging
        # ========================================================
        
        # Create subscription and publishers
        self.sub = self.create_subscription(PointCloud2, self.raw_topic, self.callback, 1)
        self.pub = self.create_publisher(PointCloud2, self.cluster_topic, 1)
        self.marker_pub = self.create_publisher(MarkerArray, self.cone_markers_topic, 1)
        
        # Startup message
        self.get_logger().info("🎯 FINE-TUNED LiDAR Cluster Node - BALANCED MODE")
        self.get_logger().info(f"Adjusted for real LiDAR clustering behavior")
        self.get_logger().info(f"Cone specs: H:{self.min_height}-{self.max_height}m, W:<{self.max_width}m")
        self.get_logger().info(f"Points: {self.cone_min_points}-{self.cone_max_points}, Density: {self.min_density}-{self.max_density}")
        self.get_logger().info(f"🔧 Key changes: width→{self.max_width}m, min_points→{self.cone_min_points}, min_height→{self.min_height}m")
        
        # Performance tracking
        self.msg_count = 0
        self.last_detection_time = 0
        
    def extract_points_fast(self, msg):
        """Point extraction with early limits"""
        try:
            points_list = []
            count = 0
            max_points = self.max_points_to_process
            
            for p in pc2.read_points(msg, skip_nans=True):
                if count >= max_points:
                    break
                    
                if hasattr(p, "__len__") and len(p) >= 3:
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    
                    dist = x*x + y*y
                    if dist < 1.0 or dist > 225.0:
                        continue
                        
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        intensity = float(p[3]) if len(p) > 3 else 0.0
                        if intensity >= self.intensity_threshold:
                            points_list.append([x, y, z])
                            count += 1
            
            return np.array(points_list, dtype=np.float32) if points_list else np.empty((0, 3), dtype=np.float32)
            
        except Exception as e:
            if self.debug_mode:
                self.get_logger().error(f"Error extracting points: {e}")
            return np.empty((0, 3), dtype=np.float32)
    
    def fast_voxel_downsample(self, points):
        """Fast voxel downsampling"""
        if len(points) == 0:
            return points
        
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        
        return points[unique_indices]
    
    def simple_cone_filter(self, points):
        """Simplified filtering for speed"""
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        # Distance filter (already pre-filtered in extraction)
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        distance_mask = (distances >= self.min_distance) & (distances <= self.max_distance)
        
        # Simple height filter
        height_mask = (points[:, 2] >= self.min_height) & (points[:, 2] <= self.max_height)
        
        return points[distance_mask & height_mask]
    
    def fast_clustering(self, points):
        """Optimized clustering for real-time performance"""
        if len(points) == 0:
            return np.array([], dtype=np.int32)
        
        labels = np.full(len(points), -1, dtype=np.int32)
        cluster_id = 0
        
        # Use KD-tree style approach for speed (simplified)
        for i in range(len(points)):
            if labels[i] != -1:
                continue
            
            # Vectorized distance calculation
            distances = np.linalg.norm(points - points[i], axis=1)
            neighbors = np.where(distances < self.eps)[0]
            
            if len(neighbors) >= self.min_points:
                labels[neighbors] = cluster_id
                cluster_id += 1
        
        return labels
    
    def enhanced_cone_validation(self, points, labels):
        """Enhanced cone validation with Formula Student specific checks"""
        if len(points) == 0:
            return []
        
        unique_clusters = np.unique(labels)
        if -1 in unique_clusters:
            unique_clusters = unique_clusters[unique_clusters != -1]
        
        cone_centroids = []
        validated_count = 0
        
        for cluster_id in unique_clusters:
            cluster_mask = (labels == cluster_id)
            cluster_points = points[cluster_mask]
            
            # Step 1: Point count validation
            if not (self.cone_min_points <= len(cluster_points) <= self.cone_max_points):
                if self.debug_mode:
                    self.get_logger().info(f"❌ Cluster {cluster_id}: Point count {len(cluster_points)} (need {self.cone_min_points}-{self.cone_max_points})")
                continue
            
            # Step 2: Dimensional analysis
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)
            
            width_x = max_bounds[0] - min_bounds[0]
            width_y = max_bounds[1] - min_bounds[1]
            height = max_bounds[2] - min_bounds[2]
            max_width = max(width_x, width_y)
            
            # Step 3: Size validation (Formula Student cone dimensions)
            if max_width > self.max_width:
                if self.debug_mode:
                    self.get_logger().info(f"❌ Cluster {cluster_id}: Too wide {max_width:.2f}m > {self.max_width}m")
                continue
                
            if not (self.min_height <= height <= self.max_height):
                if self.debug_mode:
                    self.get_logger().info(f"❌ Cluster {cluster_id}: Bad height {height:.2f}m (need {self.min_height}-{self.max_height}m)")
                continue
            
            # Step 4: Density validation
            volume = max(width_x * width_y * height, self.min_volume)
            density = len(cluster_points) / volume
            
            if not (self.min_density <= density <= self.max_density):
                if self.debug_mode:
                    self.get_logger().info(f"❌ Cluster {cluster_id}: Bad density {density:.1f} (need {self.min_density}-{self.max_density})")
                continue
            
            # Step 5: Aspect ratio validation (prevent very flat or very tall objects)
            aspect_ratio = height / max(max_width, 0.01)
            if aspect_ratio > self.max_aspect_ratio:
                if self.debug_mode:
                    self.get_logger().info(f"❌ Cluster {cluster_id}: Bad aspect ratio {aspect_ratio:.1f} > {self.max_aspect_ratio}")
                continue
            
            # Step 6: Shape validation (cone-like properties)
            if width_x < 0.05 or width_y < 0.05:
                if self.debug_mode:
                    self.get_logger().info(f"❌ Cluster {cluster_id}: Too thin - X:{width_x:.2f}m Y:{width_y:.2f}m")
                continue
            
            # If all checks pass, it's likely a cone
            centroid = np.mean(cluster_points, axis=0)
            cone_centroids.append(centroid)
            validated_count += 1
            
            if self.debug_mode:
                self.get_logger().info(f"✅ Cluster {cluster_id}: VALID CONE - {len(cluster_points)}pts, "
                                     f"w={max_width:.2f}m, h={height:.2f}m, d={density:.1f}, ar={aspect_ratio:.1f}")
        
        if self.debug_mode:
            total_clusters = len(unique_clusters)
            self.get_logger().info(f"🎯 VALIDATION: {validated_count}/{total_clusters} clusters passed strict filtering")
        
        return cone_centroids
    
    def callback(self, msg):
        """Streamlined callback for real-time performance"""
        try:
            self.msg_count += 1
            
            start_time = self.get_clock().now().nanoseconds / 1e9
            
            points = self.extract_points_fast(msg)
            
            if len(points) == 0:
                self.publish_empty_markers(msg.header.stamp)
                return
            
            points = self.fast_voxel_downsample(points)
            points = self.simple_cone_filter(points)
            
            if len(points) < 10:
                self.publish_empty_markers(msg.header.stamp)
                return
            
            labels = self.fast_clustering(points)
            cone_centroids = self.enhanced_cone_validation(points, labels)
            
            if len(cone_centroids) > 0:
                current_time = start_time
                time_since_last = current_time - self.last_detection_time
                self.last_detection_time = current_time
                
                self.get_logger().info(f"🎯 PRECISION: {len(cone_centroids)} validated cones (Δt: {time_since_last:.1f}s)")
            elif self.debug_mode and self.msg_count % 10 == 0:
                self.get_logger().info("🔍 No cones passed strict validation criteria")
            
            self.publish_fast_markers(cone_centroids, msg.header.stamp)
            
            if self.enable_cluster_publishing and self.msg_count % 3 == 0:
                self.publish_simple_clusters(points, labels, msg.header)
            
        except Exception as e:
            self.get_logger().error(f"Error in real-time callback: {e}")
    
    def publish_fast_markers(self, cone_centroids, stamp):
        """Marker publishing for real-time updates"""
        try:
            marker_array = MarkerArray()
            
            ns = f"realtime_cones_{self.msg_count % 3}"
            
            for i, centroid in enumerate(cone_centroids):
                marker = Marker()
                marker.header.frame_id = "rslidar"
                marker.header.stamp = stamp
                marker.ns = ns
                marker.id = i
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                marker.pose.position.x = float(centroid[0])
                marker.pose.position.y = float(centroid[1])
                marker.pose.position.z = float(centroid[2]) + 0.15
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.25
                marker.scale.y = 0.25
                marker.scale.z = 0.4
                
                marker.color.r = 1.0
                marker.color.g = 0.6
                marker.color.b = 0.0
                marker.color.a = 0.9
                
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = self.marker_lifetime_ns
                
                marker_array.markers.append(marker)
            
            for old_id in range(3):
                old_ns = f"realtime_cones_{old_id}"
                if old_ns != ns:
                    clear_marker = Marker()
                    clear_marker.header.frame_id = "rslidar"
                    clear_marker.header.stamp = stamp
                    clear_marker.ns = old_ns
                    clear_marker.action = Marker.DELETEALL
                    marker_array.markers.append(clear_marker)
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing fast markers: {e}")
    
    def publish_empty_markers(self, stamp):
        """Quickly clear all markers when no cones detected"""
        try:
            marker_array = MarkerArray()
            
            for ns_id in range(3):
                clear_marker = Marker()
                clear_marker.header.frame_id = "rslidar"
                clear_marker.header.stamp = stamp
                clear_marker.ns = f"realtime_cones_{ns_id}"
                clear_marker.action = Marker.DELETEALL
                marker_array.markers.append(clear_marker)
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            pass
    
    def publish_simple_clusters(self, points, labels, header):
        """Simplified cluster publishing if needed"""
        try:
            if len(points) == 0:
                return
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='cluster_id', offset=12, datatype=PointField.INT32, count=1)
            ]
            
            cloud_data = []
            for i, point in enumerate(points):
                cloud_data.append([point[0], point[1], point[2], labels[i]])
            
            clustered_cloud = pc2.create_cloud(header, fields, cloud_data)
            self.pub.publish(clustered_cloud)
            
        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = LidarClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
