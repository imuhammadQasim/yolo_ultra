# YOLO Ultra Custom Object Detection & Tracking Pipeline

An end-to-end computer vision project utilizing **YOLO Ultra** and **OpenCV (cv2)** to train custom models on **Roboflow** datasets. This repository features comprehensive training visualizations, real-time object tracking, and complete pipeline deployment scripts.

##  Key Features

*   **Custom Model Training:** End-to-end training pipeline configured for specialized Roboflow datasets.
*   **Real-Time Tracking & Inference:** Live object tracking and counting utilizing OpenCV (`cv2`) video streams.
*   **Deep Visualizations:** Complete evaluation metrics including confusion matrices, precision-recall curves, and training loss plots.
*   **Optimized Inference:** High-performance detection scripts compatible with both image batches and live webcam feeds.

##  Tech Stack

*   **Core Model:** YOLO Ultra
*   **Image Processing:** OpenCV (`cv2`)
*   **Data Sourcing:** Roboflow (Custom Datasets)
*   **Language:** Python 3.10+

##  Repository Structure

```text
├── data/               # Dataset configurations and sample images
├── models/             # Exported custom weights (.pt files)
├── src/
│   ├── train.py        # Custom model training script
│   ├── inference.py    # Static image/video detection script
│   └── live_track.py   # Real-time cv2 webcam tracking script
├── utils/              # Visualization and helper functions
├── requirements.txt    # Dependencies
└── README.md
```

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com
cd yolo_ultra
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Live Tracking
To start the real-time object tracking pipeline using your webcam and trained model:
```bash
python src/live_track.py --weights models/best.pt --source 0
```

##  Visualizations & Results
The project automatically logs training progress. Upon completing training, you can find the performance graphs, boundary box overlays, and tracking statistics saved directly into the `runs/` or `utils/` directories.