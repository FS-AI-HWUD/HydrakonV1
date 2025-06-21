"""
Hydrakon VCone (Visual Cone) Tracker
Fixed version without bluish tint
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
        
        self.bridge = CvBridge()
        self.raw_feed_pub = self.create_publisher(Image, '/zed2i/raw_feed', 5)
        self.cone_detections_pub = self.create_publisher(Image, '/zed2i/cone_detections', 5)
        self.detections_data_pub = self.create_publisher(Detection2DArray, '/zed2i/detections_data', 10)
        
        # INCREASED RESOLUTION for Foxglove
        self.stream_resolution = (640, 360)  # Increased from (320, 180)
        self.display_resolution = (1280, 720)
        self.jpeg_quality = 60  # Increased quality from 40
        self.frame_skip_counter = 0
        self.publish_image_every_n_frames = 3  # Faster publishing (was 4)
        self.publish_raw_every_n_frames = 6   # Faster publishing (was 8)
        
        self.fps_history = deque(maxlen=10)
        self.inference_history = deque(maxlen=10)
        
        self.cone_counts = defaultdict(int)
        self.total_detections = 0
        self.session_start = time.time()
        
        # Fixed color values for proper BGR display
        self.cone_types = {
            0: {'name': 'Yellow', 'color': (0, 255, 255), 'symbol': '🟡'},      # Yellow in BGR
            1: {'name': 'Blue', 'color': (255, 0, 0), 'symbol': '🔵'},         # Blue in BGR (FIXED)
            2: {'name': 'Orange', 'color': (0, 165, 255), 'symbol': '🟠'},     # Orange in BGR
            3: {'name': 'Large Orange', 'color': (0, 100, 255), 'symbol': '🟠'}, # Large Orange in BGR
            4: {'name': 'Unknown', 'color': (128, 128, 128), 'symbol': '⚪'}   # Gray in BGR
        }
        
    def initialize_model(self):
        """Load TensorRT model"""
        self.get_logger().info("Loading TensorRT model...")
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info("Hydrakon VCone Tracker initialized")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize ZED camera"""
        self.get_logger().info("Initializing ZED camera...")
        self.zed = sl.Camera()
        
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.sdk_verbose = 1
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Camera initialization failed: {err}")
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
            
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)
        
        return detection_array

    def compress_image(self, image, quality=60):
        """Compression for streaming - INCREASED QUALITY"""
        try:
            # INCREASED minimum resolution
            if image.shape[0] > 360:
                small_image = cv2.resize(image, self.stream_resolution)  # Now (640, 360)
            else:
                small_image = image
                
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', small_image, encode_param)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            return decoded
        except Exception as e:
            self.get_logger().debug(f"Compression failed: {e}")
            return image

    def publish_optimized_images(self, frame, display_frame, timestamp, detections):
        """Publish images with smart optimization - FIXED COLOR CONVERSION"""
        self.frame_skip_counter += 1
        
        if self.frame_skip_counter % self.publish_raw_every_n_frames == 0:
            try:
                # INCREASED RESOLUTION: Now using stream_resolution (640x360)
                raw_frame_small = cv2.resize(frame, self.stream_resolution)
                # FIXED: Proper RGB to BGR conversion for ROS
                raw_frame_bgr = cv2.cvtColor(raw_frame_small, cv2.COLOR_RGB2BGR)
                
                compressed_raw = self.compress_image(raw_frame_bgr, self.jpeg_quality)
                
                raw_msg = self.bridge.cv2_to_imgmsg(compressed_raw, encoding='bgr8')
                raw_msg.header.stamp = timestamp
                raw_msg.header.frame_id = "zed2i_left_camera_frame"
                self.raw_feed_pub.publish(raw_msg)
                
            except Exception as e:
                self.get_logger().debug(f"Failed to publish raw image: {e}")
        
        if detections and self.frame_skip_counter % self.publish_image_every_n_frames == 0:
            try:
                # INCREASED RESOLUTION: Now using stream_resolution (640x360)
                detection_frame_small = cv2.resize(display_frame, self.stream_resolution)
                # FIXED: Proper RGB to BGR conversion for ROS
                detection_frame_bgr = cv2.cvtColor(detection_frame_small, cv2.COLOR_RGB2BGR)
                
                # HIGHER QUALITY for detection images
                compressed_detection = self.compress_image(detection_frame_bgr, self.jpeg_quality + 10)
                
                cone_detections_msg = self.bridge.cv2_to_imgmsg(compressed_detection, encoding='bgr8')
                cone_detections_msg.header.stamp = timestamp
                cone_detections_msg.header.frame_id = "zed2i_left_camera_frame"
                self.cone_detections_pub.publish(cone_detections_msg)
                
            except Exception as e:
                self.get_logger().debug(f"Failed to publish cone detections: {e}")

    def process_detections(self, results, display_frame):
        """Process YOLO detections - FIXED for RGB frame processing"""
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
                    
                    # FIXED: Convert BGR color to RGB for drawing on RGB frame
                    color_bgr = cone_info['color']
                    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR to RGB conversion
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_rgb, 2)
                    
                    label = f"{cone_info['name']} {confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 1)
        
        for cone_name, count in current_cones.items():
            self.cone_counts[cone_name] += count
            self.total_detections += count
        
        return current_cones, detections
    
    def run(self):
        """Main processing loop - ROS topics only, no display window"""
        if not self.initialize_model():
            return
        
        if not self.initialize_camera():
            return
        
        runtime_parameters = sl.RuntimeParameters()
        image = sl.Mat()
        
        self.get_logger().info("Hydrakon VCone Tracker ACTIVE - ROS TOPICS ONLY")
        self.get_logger().info(f"Stream resolution: 640x360 (INCREASED QUALITY)")
        self.get_logger().info(f"Detection stream rate: {30/self.publish_image_every_n_frames:.1f} FPS")
        self.get_logger().info(f"Raw stream rate: {30/self.publish_raw_every_n_frames:.1f} FPS")
        self.get_logger().info("Raw feed always streams, detection images only when cones found!")
        self.get_logger().info("No local display - check Foxglove for visualization")
        self.get_logger().info("Press Ctrl+C to quit")
        
        frame_count = 0
        
        try:
            while rclpy.ok():
                loop_start = time.time()
                
                if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve RGB image
                    self.zed.retrieve_image(image, sl.VIEW.LEFT)
                    
                    frame = image.get_data()
                    # FIXED: Proper RGBA to RGB conversion (this was causing the bluish tint)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    
                    timestamp = self.get_clock().now().to_msg()
                    
                    input_frame = cv2.resize(frame, (640, 640))
                    display_frame = cv2.resize(frame, self.display_resolution)
                    
                    inference_start = time.time()
                    results = self.model(input_frame, verbose=False, conf=0.25, iou=0.7)
                    inference_time = (time.time() - inference_start) * 1000
                    
                    self.inference_history.append(inference_time)
                    
                    current_cones, detections = self.process_detections(results, display_frame)
                    
                    if detections:
                        detection_msg = self.create_detection_message(detections, timestamp)
                        self.detections_data_pub.publish(detection_msg)
                        
                        if frame_count % 30 == 0:
                            cone_summary = ", ".join([f"{name}: {count}" for name, count in current_cones.items() if count > 0])
                            self.get_logger().info(f"Detected: {cone_summary}")
                    
                    self.publish_optimized_images(frame, display_frame, timestamp, detections)
                    
                    loop_time = time.time() - loop_start
                    fps = 1.0 / loop_time if loop_time > 0 else 0
                    self.fps_history.append(fps)
                    
                    if frame_count % 300 == 0 and frame_count > 0:
                        avg_fps = np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
                        avg_inference = np.mean(self.inference_history) if len(self.inference_history) > 0 else 0
                        self.get_logger().info(f"Performance: {avg_fps:.1f} FPS, {avg_inference:.1f}ms inference, {self.total_detections} total detections")
                    
                    frame_count += 1
                
                rclpy.spin_once(self, timeout_sec=0)
        
        except KeyboardInterrupt:
            self.get_logger().info("Stopping...")
        
        finally:
            # Cleanup
            self.get_logger().info(f"PERFORMANCE SUMMARY")
            self.get_logger().info(f"Frames processed: {frame_count}")
            
            if len(self.fps_history) > 0:
                self.get_logger().info(f"Average FPS: {np.mean(self.fps_history):.1f}")
            else:
                self.get_logger().info("Average FPS: N/A")
                
            if len(self.inference_history) > 0:
                self.get_logger().info(f"Average Inference: {np.mean(self.inference_history):.1f}ms")
            else:
                self.get_logger().info("Average Inference: N/A")
                
            self.get_logger().info(f"Total Detections: {self.total_detections}")
            
            self.zed.close()

def main():
    model_path = "/home/dalek/Documents/Zed_2i/cone/best_trt_fp16_640.engine"
    
    print("HYDRAKON VCONE TRACKER")
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