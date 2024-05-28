
AI-Powered Tennis Match Analysis system with YOLO, PyTorch, and Key Point Extraction
# NetGenius: AI-Powered Tennis Match Analysis System

NetGenius is an advanced AI-powered system designed to analyze tennis matches by detecting and tracking players and the tennis ball in videos. It utilizes state-of-the-art machine learning techniques, including YOLO for object detection and CNNs for court keypoint extraction, to provide comprehensive insights into player performance.

## Features

- **Player and Ball Detection**: Detects and tracks players and the tennis ball using YOLO.
- **Court Keypoint Extraction**: Extracts keypoints of the tennis court using CNNs.
- **Player and Ball Statistics**: Calculates various statistics such as player speed, ball shot speed, and the number of shots.
- **Visual Output**: Generates annotated videos with detailed visualizations of detections and statistics.

## Project Structure

The project is organized into several stages:

1. **Input Pipeline**: Loads and preprocesses the input video.
2. **Player and Ball Detection**: Detects players and the ball in video frames.
3. **Courtline Detection**: Identifies court lines and keypoints.
4. **Minicourt Construction**: Builds a scaled-down representation of the court for easier analysis.
5. **Player Stats Calculation**: Computes various player and ball statistics.
6. **Output Drawing**: Annotates the video with detected objects and statistics.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NetGenius.git
   cd NetGenius
