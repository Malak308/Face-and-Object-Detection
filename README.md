# Computer Vision Application - 5-in-1 System

A comprehensive computer vision application featuring 5 major CV topics for practical exam preparation.

## 🎯 Features

### 1. Object Detection (YOLOv8)
- Detects 80 different object classes from COCO dataset
- Real-time detection with adjustable confidence threshold (0.15)
- Visual bounding boxes with class names and confidence scores
- Perfect for detecting everyday objects

### 2. Face & Emotion Detection (DeepFace)
- Detects faces in images
- Recognizes 7 emotions: happy, sad, angry, surprise, fear, disgust, neutral
- **Webcam support** for real-time emotion detection
- Shows confidence scores for each detected emotion
- Optimized for speed with OpenCV backend

### 3. Image Classification (ResNet50)
- Classifies images into 1000 ImageNet categories
- Top-5 predictions with confidence scores
- Visual confidence bars on output
- Pre-trained ResNet50 model

### 4. Image Segmentation
- Semantic segmentation to identify different regions
- Color-based region detection
- Side-by-side comparison with original
- Counts and visualizes detected segments

### 5. Traffic Sign Recognition
- Detects and classifies traffic signs
- Dual approach: YOLO + Color-Shape analysis
- Identifies sign types: prohibitory, warning, mandatory, guide signs
- Handles multiple signs in one image

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Malak308/Face-and-Object-Detection.git
cd Computer_Vision_proj
```

### 2. Create virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

The YOLOv8 model (`yolov8n.pt`) will be downloaded automatically on first run.

## 🚀 Usage

### Main GUI Application (Recommended)
Run the comprehensive GUI with all 5 features:
```bash
python gui_app.py
```

**GUI Features:**
- Select from 5 detection modes via radio buttons
- Upload images for analysis
- Webcam button for emotion detection (appears in Face mode)
- Side-by-side input/output comparison
- Detailed detection results with statistics
- Status log for tracking progress

### Standalone Scripts

#### Object Detection
```bash
python object_detection.py
```
Place images in `input_images/objects/`, results in `output_images/objects/`

#### Face & Emotion Detection
```bash
python face_emotion_detection.py
```
Place images in `input_images/faces/`, results in `output_images/emotions/`

#### Webcam Emotion Detection
```bash
python webcam_emotion_detection.py
```
**Controls:**
- Press 'q' to quit
- Press 's' to save current frame
- Press 'p' to pause/resume

#### Image Classification
```bash
python image_classification.py
```
Place images in `input_images/classification/`, results in `output_images/classification/`

#### Image Segmentation
```bash
python image_segmentation.py
```
Place images in `input_images/segmentation/`, results in `output_images/segmentation/`

#### Traffic Sign Recognition
```bash
python traffic_sign_recognition.py
```
Place images in `input_images/traffic_signs/`, results in `output_images/traffic_signs/`

## 📁 Project Structure

```
Computer_Vision_proj/
├── gui_app.py                      # Main GUI (5 modes + webcam)
├── object_detection.py             # Standalone object detection
├── face_emotion_detection.py       # Standalone face/emotion detection
├── webcam_emotion_detection.py     # Real-time webcam emotions
├── image_classification.py         # Image classification (ResNet50)
├── image_segmentation.py           # Image segmentation
├── traffic_sign_recognition.py     # Traffic sign recognition
├── requirements.txt                # Python dependencies
├── yolov8n.pt                     # YOLOv8 model (auto-downloaded)
├── README.md                       # This file
├── PROJECT_SUMMARY.md              # Technical documentation
├── input_images/
│   ├── objects/                   # Object detection input
│   ├── faces/                     # Face/emotion input
│   ├── classification/            # Classification input
│   ├── segmentation/              # Segmentation input
│   └── traffic_signs/             # Traffic sign input
└── output_images/
    ├── objects/                   # Object detection results
    ├── emotions/                  # Face/emotion results
    ├── classification/            # Classification results
    ├── segmentation/              # Segmentation results
    └── traffic_signs/             # Traffic sign results
```

## 💻 Requirements

- **Python**: 3.9+ (tested on 3.9.13)
- **OpenCV**: 4.8.0+
- **YOLOv8**: Ultralytics 8.0.0+
- **DeepFace**: 0.0.79+
- **TensorFlow**: 2.15.0+ (for classification)
- **PyTorch**: 2.0.0+ (for YOLO)
- **NumPy**: 1.24.0 - 1.26.x (not 2.x, incompatible with DeepFace)
- **Pillow**: 10.0.0+

All dependencies are in `requirements.txt`.

## 🎨 GUI Overview

### Dark Theme Interface
- **Colors**: Dark background (#1a1a1a), cyan accent (#00d4ff), green success (#00ff88)
- **Fonts**: Segoe UI for UI elements, Consolas for code/results
- **Window Size**: 1500x900 optimized for display

### Layout
- **Left Panel** (380px): Mode selection, upload button, webcam button (emotion mode), status log
- **Right Panel**: 
  - Top: Input/output image comparison (500px height)
  - Bottom: Detection details (expandable, 12pt Consolas font)

### Detection Modes
1. 🎯 Object Detection (80 classes)
2. 😊 Face & Emotion Detection
3. 🖼️ Image Classification (1000 classes)
4. 🎨 Image Segmentation
5. 🚦 Traffic Sign Recognition

## 📊 Performance Tips

### Object Detection
- Confidence threshold: 0.15 (adjustable)
- Works best with everyday objects from COCO dataset
- Clear lighting and multiple objects recommended

### Face/Emotion Detection
- Optimized for speed: emotion-only analysis
- Use front-facing, well-lit faces
- Webcam: analyzes every 10 frames for real-time performance
- OpenCV backend for fast face detection

### Image Classification
- ResNet50 pre-trained on ImageNet
- Top-5 predictions shown
- Best with common objects/animals

### Image Segmentation
- Color-based region detection
- Works best with distinct regions
- Side-by-side comparison output

### Traffic Sign Recognition
- Dual approach: YOLO + color-shape analysis
- Best with clear, unobstructed signs
- Handles red, blue, yellow, green signs

## 🎓 Educational Purpose

This project covers all major Computer Vision topics for practical exams:

1. ✅ **Object Detection** - YOLO-based multi-object detection
2. ✅ **Face Recognition** - Face detection + emotion classification
3. ✅ **Image Classification** - CNN-based category prediction
4. ✅ **Image Segmentation** - Region-based segmentation
5. ✅ **Specialized Detection** - Traffic sign recognition

**Bonus Feature**: Real-time webcam emotion detection with live visualization!

## 🐛 Troubleshooting

### NumPy Version Error
If you get NumPy compatibility errors with DeepFace:
```bash
pip uninstall numpy
pip install "numpy>=1.24.0,<2.0.0"
```

### Webcam Not Working
- Check camera permissions
- Try different camera index: `start_webcam(camera_index=1)`
- Ensure no other app is using the webcam

### Model Download Issues
- YOLOv8 downloads automatically (~6.2MB)
- ResNet50 downloads from Keras (~98MB)
- Check internet connection

### Slow Emotion Detection
- Already optimized (emotion only, opencv backend)
- First run initializes models (slower)
- Subsequent runs are much faster

## 📝 Credits

- **YOLOv8**: Ultralytics
- **DeepFace**: serengil
- **ResNet50**: Keras/TensorFlow
- **COCO Dataset**: Common Objects in Context

## 📄 License

Educational project for Computer Vision practical exam preparation.

## 🤝 Contributing

This is a student project. Feel free to fork and customize for your own learning!
