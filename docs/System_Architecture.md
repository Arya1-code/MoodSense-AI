# MoodSense - System Architecture

## Overview

MoodSense is a real-time emotion detection system that combines
face detection, emotion classification, and mood analytics.

## Architecture

Input Image / Webcam
        |
        v
YOLOv8 Face Detection
        |
        v
Face Cropping
        |
        v
EfficientNet Emotion Classifier
        |
        v
Emotion Prediction
        |
        +----------------+
        |                |
        v                v
Mood Analytics    Recommendations
        |
        v
React Dashboard

## Components

### Module 1: Face Detection
- YOLOv8
- Multi-face support
- Real-time inference

### Module 2: Emotion Recognition
- EfficientNet-B0
- 7 Emotion Classes
- Confidence Scores

### Module 3: Analytics
- Emotion trends
- Session statistics
- Reports

### Module 4: Frontend
- React
- Real-time visualization

### Module 5: Backend
- FastAPI
- Model serving