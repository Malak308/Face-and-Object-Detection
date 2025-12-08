# Quick Start Guide - Computer Vision Application

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
cd "m:\Coding projects\Computer_Vision _proj"
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
python gui_app.py
```

### Step 3: Start Detecting!
1. Select a detection mode (radio buttons on left)
2. Click "Choose Image" to upload
3. Click "RUN DETECTION" to analyze
4. View results on the right side

---

## 📋 All 5 Detection Modes

### 1️⃣ Object Detection
**What it does:** Finds and labels objects in images (80 classes)

**Best for:** 
- Everyday objects (cars, people, animals)
- Multiple objects in one image
- General scene understanding

**Try it with:** Photos of streets, rooms, outdoor scenes

---

### 2️⃣ Face & Emotion Detection
**What it does:** Detects faces and identifies emotions

**Best for:**
- Portraits and selfies
- Emotion analysis (happy, sad, angry, etc.)
- Multiple faces in group photos

**Bonus:** Click **📹 Use Webcam** for real-time detection!

**Webcam Controls:**
- Press `q` to quit
- Press `s` to save screenshot
- Press `p` to pause/resume

**Try it with:** Portrait photos, group photos, facial expressions

---

### 3️⃣ Image Classification
**What it does:** Identifies what's in the image (1000 categories)

**Best for:**
- Single main subject images
- Animals, objects, food
- Getting top-5 predictions

**Try it with:** Photos of single animals, objects, or scenes

---

### 4️⃣ Image Segmentation
**What it does:** Separates image into different regions

**Best for:**
- Understanding image structure
- Identifying different areas
- Visual region analysis

**Output:** Side-by-side comparison with original

**Try it with:** Images with distinct regions/colors

---

### 5️⃣ Traffic Sign Recognition
**What it does:** Detects and classifies traffic signs

**Best for:**
- Stop signs, warning signs
- Speed limit signs
- Road/traffic scenes

**Try it with:** Photos of streets with traffic signs

---

## 📁 Where to Put Images

The application auto-organizes your images:

```
input_images/
├── objects/          ← Object detection images
├── faces/            ← Face/emotion images
├── classification/   ← Classification images
├── segmentation/     ← Segmentation images
└── traffic_signs/    ← Traffic sign images

output_images/
├── objects/          ← Object detection results
├── emotions/         ← Face/emotion results
├── classification/   ← Classification results
├── segmentation/     ← Segmentation results
└── traffic_signs/    ← Traffic sign results
```

**Note:** You can upload from anywhere using the GUI, but standalone scripts use these folders.

---

## 💡 Tips for Best Results

### General Tips
✓ Use clear, well-lit images  
✓ Higher resolution = better results  
✓ Avoid heavily cropped or blurry images

### Object Detection
✓ Confidence threshold: 0.15 (very sensitive)  
✓ Works with 80 COCO classes  
✓ Multiple objects are fine

### Face/Emotion Detection
✓ Front-facing faces work best  
✓ Good lighting is crucial  
✓ Optimized for speed (emotion only)  
✓ Webcam analyzes every 10 frames

### Image Classification
✓ Center the main subject  
✓ Single object per image works best  
✓ Uses ImageNet 1000 classes

### Image Segmentation
✓ Images with distinct colors/regions  
✓ Clear boundaries help  
✓ Output shows original + segmented side-by-side

### Traffic Sign Recognition
✓ Clear, unobstructed signs  
✓ Good contrast and lighting  
✓ Dual method: YOLO + color-shape

---

## 🎨 Understanding the GUI

### Left Panel (Controls)
- **Upload Image**: Choose your image file
- **Detection Mode**: Select 1 of 5 options
- **Webcam Button**: Appears in Face mode for real-time detection
- **Run Detection**: Starts the analysis
- **Status Log**: Shows progress and messages

### Right Panel (Results)
- **Input Image**: Your uploaded image (400x450px)
- **Output Image**: Detection results (600x500px)
- **Detection Details**: Comprehensive results (12pt font, scrollable)

### Color Scheme
- **Cyan (#00d4ff)**: Accent color, main buttons
- **Green (#00ff88)**: Success indicators, upload button
- **Dark (#1a1a1a)**: Background
- **White**: Text

---

## 🐛 Common Issues

### "No module named 'tensorflow'"
**Solution:**
```bash
pip install tensorflow>=2.15.0
```

### "NumPy version error"
**Solution:**
```bash
pip uninstall numpy
pip install "numpy>=1.24.0,<2.0.0"
```

### Webcam not opening
**Solution:**
- Check if another app is using the camera
- Try different camera: edit `camera_index=0` to `camera_index=1`
- Check camera permissions in Windows Settings

### Slow first detection
**Answer:** This is normal! Models load on first run:
- YOLOv8: ~6MB download
- ResNet50: ~98MB download
- DeepFace: Initializes face detector

Subsequent runs are much faster!

### GUI not opening
**Solution:**
```bash
# Make sure you're in the correct directory
cd "m:\Coding projects\Computer_Vision _proj"

# Activate virtual environment
.venv\Scripts\activate

# Run GUI
python gui_app.py
```

---

## 📊 What You'll See

### Object Detection Output
- Bounding boxes around objects
- Class labels (e.g., "PERSON", "CAR")
- Confidence percentages
- Object count summary

### Face/Emotion Detection Output
- Green rectangles around faces
- Dominant emotion (uppercase)
- Confidence score
- All emotion percentages with bars

### Classification Output
- Original image with overlay
- Top-5 predictions
- Confidence bars for each
- Color-coded confidence

### Segmentation Output
- Original on left
- Segmented on right
- Different colors for regions
- Segment count

### Traffic Sign Output
- Boxes around signs
- Sign type labels
- Detection method (YOLO/Shape)
- Confidence scores

---

## 🎓 Learning Resources

### What Each Mode Teaches

**Object Detection:**
- Multi-object localization
- YOLO architecture
- Real-time detection

**Face/Emotion:**
- Face detection algorithms
- Emotion classification
- Real-time video processing

**Classification:**
- CNN architectures (ResNet)
- Transfer learning
- ImageNet dataset

**Segmentation:**
- Region-based analysis
- Color segmentation
- Image structure

**Traffic Signs:**
- Specialized detection
- Color-shape analysis
- Dual-method approach

---

## ⚡ Performance Expectations

| Mode | Speed | Accuracy | Notes |
|------|-------|----------|-------|
| Object Detection | Fast | High | YOLOv8n (~6MB) |
| Face/Emotion | Fast | High | Optimized, emotion-only |
| Classification | Fast | High | ResNet50 pretrained |
| Segmentation | Medium | Medium | Color-based method |
| Traffic Signs | Fast | Medium | Dual approach |
| Webcam | Real-time | High | 10-frame interval |

---

## 🎯 Exam Preparation

This project covers all major CV topics:

✅ **Object Detection** - YOLO, bounding boxes, multi-object  
✅ **Face Recognition** - Face detection, emotion classification  
✅ **Image Classification** - CNN, transfer learning, ImageNet  
✅ **Image Segmentation** - Region detection, color segmentation  
✅ **Specialized Detection** - Traffic signs, shape/color analysis

**Plus:** Real-time webcam processing, GUI development, file I/O

---

## 📞 Need Help?

1. Check the main **README.md** for detailed documentation
2. Review **PROJECT_SUMMARY.md** for technical details
3. Check error logs in the Status Log (left panel)
4. Ensure all dependencies are installed: `pip list`

---

## 🎉 Have Fun!

This is a learning project. Experiment with different images, try all modes, and understand what each technique can do.

**Remember:** First-time model loading takes longer. Be patient! ⏳

---

**Ready?** Run `python gui_app.py` and start detecting! 🚀
