# Project Goal

**Hybrid 3D Perception Pipeline for Autonomous Driving using Waymo Open Dataset**

Process a Waymo LiDAR sequence frame-by-frame using:

* LiDAR preprocessing
* Ground plane removal (RANSAC)
* Point cloud clustering
* Transformer-based 3D object detection (pretrained)
* Projection onto the front camera
* Visualization
* Runtime profiling

---

# Technology Stack

* Python
* PyTorch
* Open3D
* NumPy
* OpenCV
* Waymo Open Dataset API
* Hugging Face / MMDetection3D / OpenPCDet (choose one)
* Matplotlib (optional)

---

# Repository Structure

```
waymo-lidar-transformer/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── configs/
│   └── config.yaml
│
├── src/
│
│   ├── dataset/
│   │      waymo_loader.py
│   │
│   ├── preprocessing/
│   │      voxel.py
│   │      ground_removal.py
│   │      filters.py
│   │
│   ├── classical/
│   │      clustering.py
│   │      bounding_boxes.py
│   │
│   ├── transformer/
│   │      detector.py
│   │
│   ├── visualization/
│   │      viewer.py
│   │      projection.py
│   │
│   ├── utils/
│   │      calibration.py
│   │      timer.py
│   │
│   └── pipeline.py
│
├── outputs/
│
├── README.md
│
└── requirements.txt
```

---

# Pipeline

```
Load frame

↓

Extract LiDAR

↓

Voxel Downsample

↓

Statistical Outlier Removal

↓

Ground Removal (RANSAC)

↓

DBSCAN Clustering

↓

Transformer Detector

↓

Project detections to front camera

↓

Visualization

↓

Save frame
```

---

# Milestone 1 — Dataset Loader

### Goal

Read a Waymo sequence.

Implement

```
WaymoLoader

↓

load_sequence()

↓

load_frame(i)

↓

get_lidar()

↓

get_front_camera()

↓

get_calibration()
```

Deliverable

```
Frame 35

LiDAR:
185,432 points

Camera:
1920×1280
```

---

# Milestone 2 — Visualization

Display

* raw point cloud

using Open3D.

Then

* front camera image

Verify calibration files can be loaded.

---

# Milestone 3 — Preprocessing

Implement

Voxel Downsampling

```
voxel_size = 0.10 m
```

Then

Statistical Outlier Removal

Then

Radius Outlier Removal

Compare results.

---

# Milestone 4 — Ground Removal

Implement

RANSAC plane fitting.

Output

```
Ground points

Non-ground points
```

Visualize

Ground

↓

green

Objects

↓

red

---

# Milestone 5 — Clustering

Implement

DBSCAN

Tune

```
eps

min_points
```

Output

```
Cluster 1

Cluster 2

Cluster 3
```

Compute

Axis-aligned bounding boxes

or

Oriented bounding boxes.

---

# Milestone 6 — Transformer

Don't train.

Load pretrained model.

Possible choices

OpenPCDet

or

MMDetection3D

or

OpenMMLab

Run inference

per frame.

Store

```
boxes

scores

labels
```

---

# Milestone 7 — Sensor Fusion

Using calibration

Project

LiDAR detections

↓

front camera

Draw

```
bounding boxes

class

confidence
```

This is probably the most impressive part.

---

# Milestone 8 — Profiling

Measure

```
Loading

Voxel

RANSAC

Clustering

Detection

Projection

Visualization
```

Print

```
Frame 15

Loading ........ 10 ms

Voxel ......... 18 ms

RANSAC ........ 15 ms

DBSCAN ........ 22 ms

Detection ...... 61 ms

Projection ..... 7 ms

Total ........ 133 ms
```

---

# Milestone 9 — Multiple Sequences

Tune parameters on

```
Scene A
```

Run unchanged on

```
Scene B

Scene C
```

Discuss

* what still works
* what fails
* why

This shows generalization.

---

# Stretch Goals (Only if Time Allows)

* Track detections across frames (e.g., with Kalman Filter + Hungarian Algorithm)
* Export the detector to ONNX and compare inference speed
* Generate a BEV (Bird's-Eye View) visualization
* Save the output as a video with side-by-side LiDAR and camera views

---

# Suggested Timeline (5 Days)

### Day 1

* Set up the project structure
* Read one Waymo sequence
* Visualize LiDAR and front camera
* Verify calibration

### Day 2

* Implement preprocessing
* Ground removal (RANSAC)
* DBSCAN clustering
* Visualize intermediate results

### Day 3

* Integrate a pretrained transformer-based detector
* Run inference on several frames
* Compare detector output with classical clusters

### Day 4

* Project detections onto the front camera
* Add runtime profiling
* Test on multiple sequences

### Day 5

* Refactor the code
* Add logging and configuration
* Create a polished README with:

  * project motivation,
  * architecture diagram,
  * sample outputs,
  * discussion of limitations and future improvements

---

## One recommendation

I would **slightly change the project title** to emphasize engineering rather than just using a transformer:

> **Hybrid 3D Perception Pipeline for Autonomous Driving: Classical LiDAR Processing with Transformer-Based 3D Object Detection**

This title immediately communicates that you understand both **classical geometric methods (RANSAC, clustering)** and **modern deep-learning-based perception**, which is exactly the combination companies like SafeAD value.
