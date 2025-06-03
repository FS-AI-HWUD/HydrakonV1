"""
Optimized Hydrakon VCone Tracker - Maximum Performance
Single-threaded for optimal CUDA performance
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pyzed.sl as sl
import time
from collections import deque, defaultdict

class HydrakonVConeTracker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.zed = None
        
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
        print("🚀 Loading TensorRT model...")
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Hydrakon VCone Tracker initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize ZED camera"""
        print("📹 Initializing ZED camera...")
        self.zed = sl.Camera()
        
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.sdk_verbose = 1
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"❌ Camera initialization failed: {err}")
            return False
        
        print("✅ Camera ready")
        return True
    
    def process_detections(self, results, display_frame):
        """Process YOLO detections - optimized version"""
        current_cones = defaultdict(int)
        
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
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), cone_info['color'], 2)
                    
                    label = f"{cone_info['name']} {confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, cone_info['color'], 1)
        
        for cone_name, count in current_cones.items():
            self.cone_counts[cone_name] += count
            self.total_detections += count
        
        return current_cones
    
    def draw_hud(self, frame, current_cones, avg_fps, avg_inference):
        """Minimal HUD for maximum performance"""
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
        """Optimized main loop - single threaded for maximum speed"""
        if not self.initialize_model():
            return
        
        if not self.initialize_camera():
            return
        
        cv2.namedWindow('Hydrakon VCone Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hydrakon VCone Tracker', 1280, 720)
        
        runtime_parameters = sl.RuntimeParameters()
        image = sl.Mat()
        
        print("Hydrakon VCone Tracker ACTIVE - OPTIMIZED")
        print("Press 'q' to quit, 'r' to reset")
        
        frame_count = 0
        
        try:
            while True:
                loop_start = time.time()
                
                if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(image, sl.VIEW.LEFT)
                    frame = image.get_data()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    
                    input_frame = cv2.resize(frame, (640, 640))
                    display_frame = cv2.resize(frame, (1280, 720))
                    
                    inference_start = time.time()
                    results = self.model(input_frame, verbose=False, conf=0.25, iou=0.7)
                    inference_time = (time.time() - inference_start) * 1000
                    
                    self.inference_history.append(inference_time)
                    
                    current_cones = self.process_detections(results, display_frame)
                    
                    loop_time = time.time() - loop_start
                    fps = 1.0 / loop_time if loop_time > 0 else 0
                    self.fps_history.append(fps)
                    
                    avg_fps = np.mean(self.fps_history)
                    avg_inference = np.mean(self.inference_history)
                    
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
        
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            # Cleanup
            print(f"\nPERFORMANCE SUMMARY")
            print(f"Frames: {frame_count}")
            print(f"Avg FPS: {np.mean(self.fps_history):.1f}")
            print(f"Avg Inference: {np.mean(self.inference_history):.1f}ms")
            print(f"Total Detections: {self.total_detections}")
            
            self.zed.close()
            cv2.destroyAllWindows()

def main():
    model_path = "/home/dalek/Documents/Zed_2i/cone/best_trt_fp16_640.engine"
    
    print("OPTIMIZED HYDRAKON VCONE TRACKER")
    print("Maximum Performance Mode")
    print("=" * 40)
    
    tracker = HydrakonVConeTracker(model_path)
    tracker.run()

if __name__ == "__main__":
    main()