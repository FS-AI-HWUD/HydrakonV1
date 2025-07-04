# LiDAR Cluster Node - Real-Time Cone Detection

This ROS 2 node performs real-time clustering of LiDAR point clouds to detect traffic cones in environments such as Formula Student driverless setups. It processes incoming LiDAR data, filters it, clusters the points, validates cluster geometry and density, and publishes the final validated cones as markers and optional clustered point clouds.

---

## 📦 Features

- Fast and efficient cone detection from LiDAR point clouds
- Point extraction with distance, intensity, and height filtering
- Fast voxel downsampling for performance
- Lightweight custom clustering algorithm
- Strict cone validation (based on size, density, aspect ratio)
- Real-time marker visualization via `visualization_msgs/MarkerArray`
- Optional publishing of clustered point cloud with labels

---

## 🛠 Dependencies

### ROS 2 packages:
- `rclpy`
- `sensor_msgs`
- `geometry_msgs`
- `std_msgs`
- `visualization_msgs`

### Python libraries:
- `numpy`
- `sensor_msgs_py`
- `ament_index_python`

---

## 🚀 Running the Node

Ensure your LiDAR is publishing to `/lidar/points`, then simply run:

```bash
ros2 run <your_package_name> lidar_cluster_node.py
```

---

## 📡 ROS Topics

| Topic                          | Type                   | Description                                    |
|--------------------------------|------------------------|------------------------------------------------|
| `/lidar/points`               | `sensor_msgs/PointCloud2` | Input LiDAR point cloud                       |
| `/perception/lidar_cluster`   | `sensor_msgs/PointCloud2` | Clustered points with cluster IDs (optional) |
| `/perception/cone_markers`    | `visualization_msgs/MarkerArray` | Validated cones as visual markers     |

---

## ⚙️ Configuration Parameters (Hardcoded)

| Parameter              | Value       | Description |
|------------------------|-------------|-------------|
| `eps`                  | 0.4         | Cluster radius |
| `min_points`           | 3           | Minimum points to form a cluster |
| `min_distance`         | 1.5 m       | Minimum distance from sensor |
| `max_distance`         | 12.0 m      | Maximum distance from sensor |
| `min_height`           | 0.10 m      | Minimum height for valid cone |
| `max_height`           | 0.50 m      | Maximum height for valid cone |
| `max_width`            | 0.45 m      | Maximum cone width |
| `cone_min_points`      | 3           | Min points in cluster for valid cone |
| `cone_max_points`      | 100         | Max points in cluster for valid cone |
| `intensity_threshold`  | 8.0         | Minimum reflectance intensity |
| `voxel_size`           | 0.12        | Downsampling voxel grid size |
| `min_density`          | 10.0        | Minimum density (points per volume) |
| `max_density`          | 600.0       | Maximum density |
| `max_aspect_ratio`     | 4.0         | Max height/width ratio |
| `min_volume`           | 0.0005      | Minimum volume used to avoid divide-by-zero |
| `marker_lifetime_ns`   | 300,000,000 | Marker lifetime in nanoseconds (0.3s) |
| `process_every_n`      | 1           | Process every nth message |
| `max_points_to_process`| 1500        | Max number of LiDAR points to process |

---

## 🎯 Validation Logic

Each cluster is validated based on:

1. **Point count** in acceptable range
2. **Bounding box dimensions** (height and width)
3. **Density** (points per volume)
4. **Aspect ratio** (height to width)
5. **Minimum width** to avoid detecting thin structures

Only validated cone clusters are published as markers.

---

## 🔧 Debugging Output

With `debug_mode = True`, logs will show:

- Reasons for invalid clusters (e.g., width, height, density issues)
- Cone detection summaries
- Total validated cones per callback

---

## 📄 License

This code is part of a larger LiDAR perception module used in robotics competitions. Licensing depends on the containing repository.
