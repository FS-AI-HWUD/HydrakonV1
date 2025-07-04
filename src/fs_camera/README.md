# Hydrakon VCone Tracker

**Hydrakon VCone (Visual Cone) Tracker** is a ROS 2 node for real-time cone detection using the ZED 2i stereo camera and a YOLO object detection model accelerated via TensorRT. The node captures RGB images from the ZED camera, performs inference using a YOLO model, and publishes raw and annotated detection streams along with structured detection data over ROS 2 topics.

---

## 📦 Features

- Real-time YOLO detection using TensorRT engine
- Fixed color conversion (no bluish tint)
- Cone type classification: Yellow, Blue, Orange, Large Orange, Unknown
- ROS 2 image publishing with resolution and JPEG quality controls
- Publishes detection bounding boxes as `vision_msgs/Detection2DArray`
- Frame skipping logic to optimize streaming and inference frequency
- No GUI window — works headlessly (for Foxglove or similar tools)

---

## 🛠 Dependencies

### ROS 2 packages:
- `rclpy`
- `sensor_msgs`
- `vision_msgs`
- `std_msgs`
- `cv_bridge`

### Python packages:
- `opencv-python`
- `numpy`
- `ultralytics`
- `pyzed.sl` (ZED SDK Python API)

### Hardware:
- **ZED 2i camera** with proper SDK and drivers installed

---

## 🚀 Running the Node

Update the `model_path` variable in the `main()` function with the absolute path to your YOLO `.engine` file (TensorRT format):

```python
model_path = "/home/dalek/Documents/Zed_2i/cone/best_trt_fp16_640.engine"
```

Then run the script:

```bash
python3 hydrakon_vcone_tracker.py
```

Ensure ROS 2 is sourced and active.

---

## 📡 ROS Topics

| Topic                        | Type                      | Description                                  |
|-----------------------------|---------------------------|----------------------------------------------|
| `/zed2i/raw_feed`           | `sensor_msgs/Image`       | Compressed RGB image from ZED camera         |
| `/zed2i/cone_detections`    | `sensor_msgs/Image`       | Annotated detection image                    |
| `/zed2i/detections_data`    | `vision_msgs/Detection2DArray` | Structured detection data             |

---

## 🎯 Cone Classes and Colors

| Class ID | Name          | Color (BGR)      | Symbol |
|----------|---------------|------------------|--------|
| 0        | Yellow        | (0, 255, 255)    | 🟡     |
| 1        | Blue          | (255, 0, 0)      | 🔵     |
| 2        | Orange        | (0, 165, 255)    | 🟠     |
| 3        | Large Orange  | (0, 100, 255)    | 🟠     |
| 4        | Unknown       | (128, 128, 128)  | ⚪     |

---

## ⚙️ Configuration Parameters (Hardcoded)

- **Stream resolution:** 640×360
- **Display resolution:** 1280×720
- **JPEG quality:** 60
- **Model confidence threshold:** 0.25
- **IOU threshold:** 0.7
- **Image publishing rate:**
  - Detections: every 3 frames
  - Raw feed: every 6 frames

---

## 📈 Performance Metrics

The script calculates:
- Inference time per frame (ms)
- Frames per second (FPS)
- Total detections
- Periodic summaries every 300 frames

All performance data is printed to the ROS logger.

---

## ❌ Troubleshooting Notes

- If the model fails to load, the node will log the error and exit.
- If the ZED camera fails to initialize, the node will exit.
- Ensure proper conversion from RGBA → RGB to prevent color artifacts (handled in code).
- Ensure your YOLO model supports 640×640 resolution as it is hardcoded.

---

## 🔚 Shutdown and Cleanup

On shutdown (Ctrl+C), the node:
- Logs final FPS and inference averages
- Reports total detections
- Closes the ZED camera safely

---

## 📄 License

This script is part of the Hydrakon project. All usage must comply with the license defined for the full project.