# Tennis Analysis System

## Demo
![Demo](output_videos/demo.gif)

## Overview
The Tennis Analysis System is an advanced computer vision project designed to extract deep insights from tennis match videos. By leveraging state-of-the-art deep learning models, this system automatically tracks players and the ball, maps their real-world coordinates to a standardized mini-court, and computes comprehensive game analytics such as player movement and shot detection.

## Key Features
* **Player Tracking**: Utilizes advanced YOLO object detection models to accurately track players across the court, actively filtering out spectators and officials through court spatial awareness.
* **Ball Tracking & Interpolation**: Identifies and tracks the tennis ball continuously, filling in missing or occluded frames with sophisticated interpolation algorithms.
* **Court Keypoint Detection**: Automatically detects structural lines and keypoints of the tennis court to understand perspective, depth, and geometry.
* **Mini-Court Projection**: Maps the real-world positions of players and the ball onto a top-down 2D mini-court representation.
* **Shot Detection**: Analyzes the ball's trajectory to accurately identify when and where shots are taken.
* **In-Depth Analytics**: Computes vital statistics such as player speed, total distance covered, and rally information frame-by-frame.
* **Stream-Based Processing**: Optimized architecture that processes video streams lazily, ensuring efficient memory usage and preventing RAM overflow during the analysis of long, high-resolution videos.

## Architecture & Modules
The system is highly modularized for maintainability and scalability:
* `main.py`: The core execution script that orchestrates the entire analysis pipeline from ingestion to rendering.
* `tracker/`: Contains `PlayerTracker` and `BallTracker` classes responsible for continuous object detection, stub caching, and movement analysis.
* `court_line_detector/`: Implements the `CourtLineDetector` to identify structural court keypoints using a custom neural network.
* `mini_court/`: Houses the `MiniCourt` representation, handling advanced coordinate transformations from a 3D camera perspective to a 2D topological mapping.
* `utils/`: A collection of robust utility modules.
    * `video_utils.py`: Stream-based video reading and saving, abstracting IO operations.
    * `draw_utils.py`: Handles all rendering and visual annotations, decoupled from the core logic for cleaner architecture.
    * `analytics_utils.py`: Performs all mathematical computations for speed, distance, and player performance.

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

## Performance Enhancements
The core pipeline operates on stream-based processing rather than loading entire videos into memory. This optimization guarantees that the application can handle extensive, high-resolution videos without encountering memory limitations. Furthermore, rendering logic has been fully decoupled from the core loop, establishing a clean, highly maintainable architecture ready for future scaling.
