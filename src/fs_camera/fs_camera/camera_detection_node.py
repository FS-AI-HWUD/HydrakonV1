#!/usr/bin/env python3
"""
Hydrakon VCone Tracker - Simple and Fast
Based on your working version with minimal, proven optimizations
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import pyzed.sl as sl
import time
from collections import deque, defaultdict

class HydrakonVConeTracker(Node):
    def __init__(self, model_path):
        super().__init__('hydrakon_vcone_tracker')
        
        self.model_path = model_path
        self.model = None
        self.zed = None
        
        # ROS2 setup
        self.bridge = CvBridge()
        self.raw_feed_pub = self.create_publisher(Image, '/zed2i/raw_feed', 10)
        self.cone_detections_pub = self.create_publisher(Image, '/zed2i/cone_detections', 10)
        self.detections_data_pub = self.create_publisher(Detection2DArray, '/zed2i/detections_data', 10)
        
        self.fps_history = deque(maxlen=10)
        self.inference_history = deque(maxlen=10)
        
        self.cone_counts = defaultdict(int)
        self.total_detections = 0
        self.session_start = time.time()
        
        self.cone_types = {
            0: {'name': 'Yellow', 'color': (0, 255, 255), 'symbol': '🟡'},
            1: {'name': 'Blue', 'color': (255, 100, 0), 'symbol': '🔵'},
            2: {'name': 'Orange', 'color': (0, 165, 255), 'symbol': '🟠'},
            3: {'name': 'Large Orange', 'color': (0, 100, 255), 'symbol': '🟠'},
            4: {'name': 'Unknown', 'color': (128, 128, 128), 'symbol': '⚪'}
        }
        
    def initialize_model(self):
        """Load TensorRT model"""
        self.get_logger().info("🚀 Loading TensorRT model...")
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info("✅ Hydrakon VCone Tracker initialized")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load model: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize ZED camera"""
        self.get_logger().info("📹 Initializing ZED camera...")
        self.zed = sl.Camera()
        
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE  # No depth needed
        init_params.sdk_verbose = 1
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"❌ Camera initialization failed: {err}")
            return False
        
        self.get_logger().info("✅ Camera ready")
        return True
    
    def create_detection_message(self, detections, timestamp):
        """Create ROS2 Detection2DArray message"""
        detection_array = Detection2DArray()
        
        header = Header()
        header.stamp = timestamp
        header.frame_id = "zed2i_left_camera_frame"
        detection_array.header = header
        
        for det in detections:
            detection = Detection2D()
            detection.header = header
            
            bbox = det['bbox']
            detection.bbox.center.position.x = float((bbox[0] + bbox[2]) / 2)
            detection.bbox.center.position.y = float((bbox[1] + bbox[3]) / 2)
            detection.bbox.size_x = float(bbox[2] - bbox[0])
            detection.bbox.size_y = float(bbox[3] - bbox[1])
            
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det['class'])
            hypothesis.hypothesis.score = float(det['confidence'])
            
            # Remove distance info since we're not using depth
            # if 'distance' in det and det['distance'] > 0:
            #     hypothesis.pose.pose.position.z = float(det['distance'])
            
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)
        
        return detection_array

    def process_detections(self, results, display_frame):
        """Process YOLO detections"""
        current_cones = defaultdict(int)
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            h_scale = display_frame.shape[0] / 640
            w_scale = display_frame.shape[1] / 640
            
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i].astype(int)
                confidence = float(conf[i])
                class_id = cls[i]
                
                x1, x2 = int(x1 * w_scale), int(x2 * w_scale)
                y1, y2 = int(y1 * h_scale), int(y2 * h_scale)
                
                if class_id in self.cone_types:
                    cone_info = self.cone_types[class_id]
                    current_cones[cone_info['name']] += 1
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': class_id
                    })
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), cone_info['color'], 2)
                    
                    # Simple label without distance
                    label = f"{cone_info['name']} {confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, cone_info['color'], 1)
        
        for cone_name, count in current_cones.items():
            self.cone_counts[cone_name] += count
            self.total_detections += count
        
        return current_cones, detections
    
    def draw_hud(self, frame, current_cones, avg_fps, avg_inference):
        """Simple HUD"""
        y = 30
        line_height = 25
        
        hud_data = [
            f"HYDRAKON VCONE TRACKER",
            f"FPS: {avg_fps:.1f} | Inference: {avg_inference:.1f}ms",
            f"Total: {self.total_detections}",
        ]
        
        for cone_name, count in current_cones.items():
            if count > 0:
                hud_data.append(f"{cone_name}: {count}")
        
        for i, text in enumerate(hud_data):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (20, y + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    def run(self):
        """Main processing loop - simple and reliable"""
        if not self.initialize_model():
            return
        
        if not self.initialize_camera():
            return
        
        cv2.namedWindow('Hydrakon VCone Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hydrakon VCone Tracker', 1280, 720)
        
        runtime_parameters = sl.RuntimeParameters()
        image = sl.Mat()
        
        self.get_logger().info("Hydrakon VCone Tracker ACTIVE")
        self.get_logger().info("Press 'q' to quit, 'r' to reset")
        
        frame_count = 0
        
        try:
            while rclpy.ok():
                loop_start = time.time()
                
                if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve RGB image only
                    self.zed.retrieve_image(image, sl.VIEW.LEFT)
                    
                    frame = image.get_data()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    
                    timestamp = self.get_clock().now().to_msg()
                    
                    input_frame = cv2.resize(frame, (640, 640))
                    display_frame = cv2.resize(frame, (1280, 720))
                    
                    inference_start = time.time()
                    results = self.model(input_frame, verbose=False, conf=0.25, iou=0.7)
                    inference_time = (time.time() - inference_start) * 1000
                    
                    self.inference_history.append(inference_time)
                    
                    current_cones, detections = self.process_detections(results, display_frame)
                    
                    if detections:
                        detection_msg = self.create_detection_message(detections, timestamp)
                        self.detections_data_pub.publish(detection_msg)
                    
                    # Publish raw feed (without annotations)
                    try:
                        raw_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        raw_msg = self.bridge.cv2_to_imgmsg(raw_frame_bgr, encoding='bgr8')
                        raw_msg.header.stamp = timestamp
                        raw_msg.header.frame_id = "zed2i_left_camera_frame"
                        self.raw_feed_pub.publish(raw_msg)
                    except Exception as e:
                        self.get_logger().error(f"Failed to publish raw image: {e}")
                    
                    # Publish cone detections (annotated feed for RViz visualization)
                    try:
                        display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                        cone_detections_msg = self.bridge.cv2_to_imgmsg(display_frame_bgr, encoding='bgr8')
                        cone_detections_msg.header.stamp = timestamp
                        cone_detections_msg.header.frame_id = "zed2i_left_camera_frame"
                        self.cone_detections_pub.publish(cone_detections_msg)
                    except Exception as e:
                        self.get_logger().error(f"Failed to publish cone detections image: {e}")
                    
                    loop_time = time.time() - loop_start
                    fps = 1.0 / loop_time if loop_time > 0 else 0
                    self.fps_history.append(fps)
                    
                    avg_fps = np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
                    avg_inference = np.mean(self.inference_history) if len(self.inference_history) > 0 else 0
                    
                    self.draw_hud(display_frame, current_cones, avg_fps, avg_inference)
                    
                    cv2.imshow('Hydrakon VCone Tracker', display_frame)
                    frame_count += 1
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.cone_counts.clear()
                    self.total_detections = 0
                    self.session_start = time.time()
                    self.get_logger().info("Statistics reset")
                
                rclpy.spin_once(self, timeout_sec=0)
        
        except KeyboardInterrupt:
            self.get_logger().info("Stopping...")
        
        finally:
            # Cleanup
            self.get_logger().info(f"PERFORMANCE SUMMARY")
            self.get_logger().info(f"Frames: {frame_count}")
            
            if len(self.fps_history) > 0:
                self.get_logger().info(f"Avg FPS: {np.mean(self.fps_history):.1f}")
            else:
                self.get_logger().info("Avg FPS: N/A")
                
            if len(self.inference_history) > 0:
                self.get_logger().info(f"Avg Inference: {np.mean(self.inference_history):.1f}ms")
            else:
                self.get_logger().info("Avg Inference: N/A")
                
            self.get_logger().info(f"Total Detections: {self.total_detections}")
            
            self.zed.close()
            cv2.destroyAllWindows()

def main():
    model_path = "/home/dalek/Documents/Zed_2i/cone/best_trt_fp16_640.engine"
    
    print("HYDRAKON VCONE TRACKER - SIMPLE AND FAST")
    print("Cone Detection + ROS2 Publishing")
    print("=" * 50)
    
    rclpy.init()
    
    try:
        tracker = HydrakonVConeTracker(model_path)
        tracker.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()