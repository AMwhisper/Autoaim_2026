# AutoAim 2026

Real-time RoboMaster Auto-Aiming System based on NVIDIA Jetson Orin Nano, YOLOv8, Kalman Filtering, and Ballistic Compensation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano-76B900)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

AutoAim 2026 is an autonomous targeting system developed for RoboMaster robots.

The system runs on NVIDIA Jetson Orin Nano and uses Daheng Galaxy industrial cameras for image acquisition. YOLOv8 is employed for armor detection, followed by PnP-based pose estimation, Kalman Filter motion prediction, and ballistic compensation to generate accurate Yaw and Pitch commands.

A lightweight web monitoring interface is also provided for real-time visualization and debugging.

---

## Features

* Real-time armor detection using YOLOv8
* TensorRT acceleration support
* PnP-based 3D pose estimation
* Kalman Filter target tracking
* Ballistic compensation
* Daheng Galaxy camera support
* Web-based monitoring dashboard
* Modular architecture
* NVIDIA Jetson deployment

---

## Pipeline

```text
Daheng Camera
      │
      ▼
YOLOv8 Detection
      │
      ▼
Armor Matching
      │
      ▼
PnP Solver
      │
      ▼
Kalman Tracker
      │
      ▼
Ballistic Solver
      │
      ▼
Yaw / Pitch Output
      │
      ▼
Robot Controller
```

---

## Project Structure

```text
RV_Autoaim_2026/
│
├── weight/
│   ├── best.pt
│   ├── best.onnx
│   ├── best_256n.engine
│   ├── best_256s.engine
│   └── best_320s.engine
│
├── autoaim.py
├── detector.py
├── camera.py
├── pnp_solver.py
├── kalman_tracker.py
├── ballistic_solver.py
├── publisher.py
├── monitor.py
├── logger.py
├── start.py
│
├── setup.py
└── README.md
```

---

## Core Modules

### detector.py

YOLOv8-based armor detection.

Supports:

* PyTorch (.pt)
* ONNX (.onnx)
* TensorRT (.engine)

---

### pnp_solver.py

Computes the 3D pose of detected armor plates using Perspective-n-Point (PnP).

Outputs:

* Target position
* Distance
* Rotation angles

---

### kalman_tracker.py

Target state estimation and prediction.

Provides:

* Position smoothing
* Velocity estimation
* Motion prediction

---

### ballistic_solver.py

Projectile trajectory compensation.

Considers:

* Bullet speed
* Gravity
* Flight time
* Camera-to-barrel offset

Outputs compensated Yaw and Pitch angles.

---

### monitor.py

Web-based monitoring interface.

Displays:

* Detection results
* Bounding boxes
* Tracking information
* Runtime statistics

Accessible from:

```text
localhost
LAN devices
```

---

## Installation

### Install Daheng SDK

Install:

* Galaxy Gige-U3 SDK
* Galaxy Python SDK

The Gige-U3 SDK must be installed before the Python SDK.

---

### Install Package

```bash
python3 -m pip install -e .
```

---

## Run

Start the system:

```bash
autoaim_start
tmux #To run the process at startup, use tmux 
```

Optional parameters can be used to configure:

* Robot type
* Team color
* Camera parameters
* Detection model
* Web monitor
  
---

## Model Formats

The project supports multiple deployment formats.

| Format          | Description                       |
| --------------- | --------------------------------- |
| PT              | PyTorch                           |
| ONNX            | Cross-platform deployment         |
| TensorRT Engine | High-performance Jetson inference |

---

## Hardware

### Compute

* NVIDIA Jetson Orin Nano 8G

### Camera

* Daheng Galaxy Industrial Camera

### Controller

* STM32-based RoboMaster Controller

---

## Future Work

* Stereo Vision Support
* Radar-Camera Fusion
* Multi-Target Prediction
* ROS2 Integration
* End-to-End Perception Pipeline

---

## Author

Jingqi Mei

Robotics • Computer Vision • Machine Learning

GitHub: https://github.com/AMwhisper
