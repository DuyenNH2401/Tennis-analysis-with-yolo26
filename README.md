# Tennis Match Analytics Pipeline

## Demo
![Demo](output_videos/demo.gif)

## Overview
The **Tennis Match Analytics Pipeline** is an end-to-end computer vision and data analytics system designed to extract deep tactical insights from tennis match videos. Going beyond standard object detection, this system integrates multiple deep learning models, advanced signal processing, and spatial transformations to automatically track players, interpolate high-speed ball trajectories, map real-world coordinates to a 2D mini-court, and detect key match events like ball hits.

This project was built to demonstrate a complete, scalable AI pipeline applicable to real-world sports analytics.

## Technical Highlights 
* **Multi-Model Pipeline**: Integrates **YOLO** for robust object tracking (players & ball) and a custom **ResNet50** model for predicting 14 specific court structural keypoints.
* **Advanced Data Handling & Signal Processing**: 
  * Implements `pandas` interpolation to handle missing frames caused by high-speed ball occlusion.
  * Utilizes time-series analysis (`rolling_mean` and `delta_y` differentiation) to automatically detect when the tennis ball is hit.
* **Domain Knowledge Heuristics**: Features custom algorithms designed to filter out spectators/officials and isolate the two active players based on distance to the court center and topological positioning.
* **Spatial Transformation (3D to 2D)**: Projects real-world video coordinates onto a top-down 2D mini-court mapping, utilizing player height references to estimate depth and scale.
* **Stream-Based Processing**: Optimized core architecture that processes video streams efficiently, avoiding memory constraints (RAM overflow) common when handling high-resolution, long-duration videos.

## Key Features
* **Player Tracking**: Utilizes advanced YOLO object detection models to accurately track players across the court.
* **Ball Tracking & Interpolation**: Identifies and tracks the tennis ball continuously, filling in missing or occluded frames with sophisticated interpolation algorithms.
* **Court Keypoint Detection**: Automatically detects structural lines and keypoints of the tennis court to understand perspective, depth, and geometry.
* **Mini-Court Projection**: Maps the real-world positions of players and the ball onto a top-down 2D mini-court representation.
* **Automated Shot Detection**: Analyzes the ball's trajectory to accurately identify when and where shots are taken.
* *(Coming Soon)* **In-Depth Analytics**: Computes vital statistics such as player speed, total distance covered, and spatial heatmaps.

## Architecture & Modules
The system is highly modularized for maintainability and scalability, adhering strictly to Object-Oriented Programming principles:
* `main.py`: The core execution script that orchestrates the entire analysis pipeline from ingestion to rendering.
* `tracker/`: Contains `PlayerTracker` and `BallTracker` classes responsible for continuous object detection, stub caching, and movement analysis.
* `court_line_detector/`: Implements the `CourtLineDetector` to identify structural court keypoints using a custom neural network.
* `mini_court/`: Houses the `MiniCourt` representation, handling advanced coordinate transformations from a 3D camera perspective to a 2D topological mapping.
* `utils/`: A collection of robust utility modules (video I/O, annotations, bounding box math).

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure you have the necessary model weights placed in the appropriate directories before executing:
* `yolo26l.pt` (Player detection - root directory)
* `models/yolo26m_best.pt` (Ball detection)
* `models/best_keypoints_model.pt` (Court keypoints detection)

## Usage

Run the analysis by executing the `main.py` script. You can specify input and output video paths using command-line arguments.

```bash
python main.py --input ./input_videos/source.mp4 --output ./output_videos/result.mp4
```

### Arguments
* `--input`: Path to the source video file (default: `./input_videos/input_video_10s_2.mp4`)
* `--output`: Path where the analyzed output video will be saved (default: `./output_videos/output_video.mp4`)

## Future Roadmap & MLOps
- [ ] Implement mathematically rigorous Homography (`cv2.findHomography`) for optimal 3D-to-2D perspective transformation.
- [ ] Add interactive UI using **Streamlit** or **Gradio** for easy video uploads and analysis viewing.
- [ ] Dockerize the entire pipeline for seamless deployment and reproducibility.
