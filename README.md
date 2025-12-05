# Computer Vision Project - Object Detection & Face Emotion Recognition

A comprehensive computer vision project featuring two powerful applications: **Object Detection with YOLOv8** and **Face Recognition with Emotion Detection**.

## Project Overview

This project implements two fundamental computer vision tasks:

### 1. **Object Detection** 
Identifies and locates multiple objects in images using YOLOv8, drawing bounding boxes with labels and confidence scores.

### 2. **Face Recognition & Emotion Detection**
Detects faces in images and analyzes emotions (happy, sad, angry, neutral, etc.) along with age and gender estimation.

## Why Object Detection?

Object detection is a crucial computer vision application with real-world uses including:
- Autonomous vehicles (detecting pedestrians, cars, traffic signs)
- Security systems (monitoring and alerting)
- Retail analytics (counting products, detecting inventory)
- Medical imaging (identifying abnormalities)

## Technical Approach

### Model Choice: YOLOv8
I chose **YOLOv8 (You Only Look Once)** because:
- **Speed**: Single-pass detection makes it very fast
- **Accuracy**: State-of-the-art performance on object detection benchmarks
- **Ease of use**: Pre-trained on COCO dataset (80 object classes)
- **Production-ready**: Widely used in real-world applications

### How It Works

1. **Image Loading**: Reads images using OpenCV
2. **Preprocessing**: YOLOv8 automatically handles image preprocessing
3. **Detection**: Model processes the image in a single forward pass
4. **Post-processing**: 
   - Non-maximum suppression removes duplicate detections
   - Filters detections by confidence threshold (default: 50%)
5. **Visualization**: Draws bounding boxes with labels and confidence scores
6. **Output**: Saves annotated images showing all detected objects

### What Can Be Detected?

The model can detect 80 different object classes from the COCO dataset, including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, boat, train
- **Animals**: dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Common Objects**: bottle, cup, fork, knife, spoon, bowl, chair, couch, bed, table
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone
- And many more...

## Project Structure

```
Computer_Vision_proj/
├── object_detection.py           # YOLOv8 object detection
├── face_emotion_detection.py     # Face & emotion detection
├── download_face_samples.py      # Download face image samples
├── requirements.txt              # Python dependencies
├── input_images/                 # Place test images here
├── output_images/                # Detection results
│   └── emotions/                 # Emotion detection results
└── README.md                     # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First run will download models (~600MB total):
- YOLOv8 model: ~6MB
- DeepFace models: ~600MB (emotion, age, gender, race)

This installs:
- `opencv-python`: Image processing
- `ultralytics`: YOLOv8 implementation
- `deepface`: Face recognition & emotion detection
- `tensorflow`: Deep learning framework
- `numpy`: Numerical operations
- `pillow`: Image handling

### 2. Add Test Images

Place your test images in the `input_images/` folder. Supported formats:
- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

### 3. Run Object Detection

```bash
python object_detection.py
```

The script will:
- Load the YOLOv8 model (downloads automatically on first run)
- Process all images in `input_images/`
- Save results to `output_images/`
- Print detection summary for each image

### 4. Run Face & Emotion Detection

```bash
# Download sample face images (optional)
python download_face_samples.py

# Run emotion detection
python face_emotion_detection.py
```

The script will:
- Detect all faces in images
- Analyze emotions for each face
- Estimate age and gender
- Save annotated results to `output_images/emotions/`

## Example Outputs

### Object Detection
```
==================================================
Image: street_scene.jpg
==================================================
Total objects detected: 5

Detected objects:
  - car: 95.23% confidence
  - person: 89.45% confidence
  - traffic light: 78.32% confidence
  - bicycle: 82.11% confidence
  - dog: 91.67% confidence

Output saved to: output_images/detected_street_scene.jpg
==================================================
```

### Face & Emotion Detection
```
==================================================
Processing: face_sample_1.jpg
==================================================
Found 1 face(s)

Face 1:
  Emotion: happy (100.0%)
  Age: ~32 years
  Gender: Woman (100.0%)
  All emotions:
    - happy: 100.0%
    - neutral: 0.0%
    - surprise: 0.0%
    - sad: 0.0%

Output saved to: output_images/emotions/emotion_face_sample_1.jpg
==================================================
```

## Key Features

- **Automatic Model Download**: YOLOv8 weights download on first run
- **Batch Processing**: Processes all images in input folder
- **Confidence Filtering**: Only shows detections above 50% confidence
- **Visual Annotations**: Colored bounding boxes with labels
- **Detailed Reports**: Prints what was detected in each image

## Customization Options

You can modify the script to adjust:
- **Confidence threshold**: Change `confidence_threshold` parameter (default: 0.5)
- **Model size**: Switch between yolov8n (nano), yolov8s (small), yolov8m (medium), etc.
- **Output format**: Modify visualization style in the code

## Performance

- **Model**: YOLOv8n (nano version)
- **Speed**: ~20-30ms per image on CPU, ~5-10ms on GPU
- **Accuracy**: High precision for common objects
- **Size**: ~6MB model file

## Learning Outcomes

This project demonstrates:
1. Using pre-trained deep learning models
2. Image processing with OpenCV
3. Real-world computer vision pipeline (load → process → visualize → save)
4. Working with modern object detection architectures
5. Practical implementation of YOLO algorithm

## Next Steps / Improvements

Potential enhancements:
- Add video processing capability
- Implement custom object training
- Add object tracking across frames
- Create a web interface
- Optimize for edge devices (Raspberry Pi, Jetson Nano)

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

**Project Created**: December 2025  
**Purpose**: Computer Vision Practical Exam
