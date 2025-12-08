"""
Computer Vision GUI Application
Supports:
- Object Detection
- Face/Emotion Detection (with webcam support)
- Image Classification
- Image Segmentation
- Traffic Sign Recognition
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageOps
import cv2
from ultralytics import YOLO
from deepface import DeepFace
from pathlib import Path
import threading
import os

# Import our CV modules
from image_classification import ImageClassifier
from image_segmentation import ImageSegmenter
from traffic_sign_recognition import TrafficSignRecognizer
from webcam_emotion_detection import WebcamEmotionDetector

class ComputerVisionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Computer Vision Application")
        self.root.geometry("1500x900")
        self.root.configure(bg="#1a1a1a")
        
        # Configure style
        self.setup_styles()
        
        # Create output directories
        self.setup_directories()
        
        # Load models (will be done on first use)
        self.yolo_model = None
        self.classifier = None
        self.segmenter = None
        self.traffic_recognizer = None
        self.webcam_detector = None
        self.current_image_path = None
        self.current_output_path = None
        self.stop_flag = False
        self.processing_thread = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_styles(self):
        """Setup color scheme and fonts"""
        self.colors = {
            'bg_dark': '#1a1a1a',
            'bg_medium': '#2d2d2d',
            'bg_light': '#3d3d3d',
            'accent': '#00d4ff',
            'success': '#00ff88',
            'text': '#ffffff',
            'text_dim': '#b0b0b0'
        }
        
        self.fonts = {
            'title': ('Segoe UI', 28, 'bold'),
            'heading': ('Segoe UI', 14, 'bold'),
            'button': ('Segoe UI', 12, 'bold'),
            'text': ('Segoe UI', 11),
            'small': ('Segoe UI', 10),
            'mono': ('Consolas', 10)
        }
    
    def setup_directories(self):
        """Create necessary directories for all 5 CV modes"""
        base_dir = Path(__file__).parent
        
        # Input folders
        self.object_input = base_dir / "input_images" / "objects"
        self.face_input = base_dir / "input_images" / "faces"
        self.classification_input = base_dir / "input_images" / "classification"
        self.segmentation_input = base_dir / "input_images" / "segmentation"
        self.traffic_input = base_dir / "input_images" / "traffic_signs"
        
        # Output folders
        self.object_output = base_dir / "output_images" / "objects"
        self.face_output = base_dir / "output_images" / "emotions"
        self.classification_output = base_dir / "output_images" / "classification"
        self.segmentation_output = base_dir / "output_images" / "segmentation"
        self.traffic_output = base_dir / "output_images" / "traffic_signs"
        
        all_dirs = [
            self.object_input, self.face_input, self.classification_input, 
            self.segmentation_input, self.traffic_input,
            self.object_output, self.face_output, self.classification_output,
            self.segmentation_output, self.traffic_output
        ]
        
        for directory in all_dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_gui(self):
        """Setup the GUI layout"""
        
        # Title
        title_frame = tk.Frame(self.root, bg=self.colors['bg_medium'], height=90)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🔍 Computer Vision Detection System",
            font=self.fonts['title'],
            bg=self.colors['bg_medium'],
            fg=self.colors['accent']
        )
        title_label.pack(expand=True, pady=(15, 5))
        
        subtitle_label = tk.Label(
            title_frame,
            text="Object Detection • Emotion Recognition • Image Classification • Segmentation • Traffic Signs",
            font=self.fonts['small'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dim']
        )
        subtitle_label.pack(pady=(0, 10))
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        # Left panel - Controls
        left_panel = tk.Frame(content_frame, bg=self.colors['bg_medium'], width=400, relief=tk.FLAT, bd=0)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Upload section
        upload_frame = tk.LabelFrame(
            left_panel,
            text="📁 Upload Image",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            padx=20,
            pady=15,
            relief=tk.FLAT,
            borderwidth=0
        )
        upload_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        self.upload_btn = tk.Button(
            upload_frame,
            text="📤 Choose Image",
            command=self.upload_image,
            font=self.fonts['button'],
            bg=self.colors['success'],
            fg="#000000",
            cursor="hand2",
            height=2,
            relief=tk.FLAT,
            activebackground="#00cc70",
            borderwidth=0
        )
        self.upload_btn.pack(fill=tk.X, pady=(0, 10))
        
        self.file_label = tk.Label(
            upload_frame,
            text="No file selected",
            font=self.fonts['small'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dim'],
            wraplength=320
        )
        self.file_label.pack(pady=8)
        
        # Detection mode selection
        mode_frame = tk.LabelFrame(
            left_panel,
            text="🎯 Detection Mode",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            padx=20,
            pady=15,
            relief=tk.FLAT,
            borderwidth=0
        )
        mode_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.mode_var = tk.StringVar(value="object")
        
        modes = [
            ("🎯 Object Detection (80 classes)", "object"),
            ("😊 Face & Emotion Detection", "face"),
            ("🖼️ Image Classification (1000 classes)", "classification"),
            ("🎨 Image Segmentation", "segmentation"),
            ("🚦 Traffic Sign Recognition", "traffic")
        ]
        
        for text, value in modes:
            radio = tk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                font=self.fonts['text'],
                bg=self.colors['bg_medium'],
                fg=self.colors['text'],
                selectcolor=self.colors['bg_light'],
                activebackground=self.colors['bg_medium'],
                activeforeground=self.colors['accent'],
                command=self.on_mode_change
            )
            radio.pack(anchor=tk.W, pady=5)
        
        # Webcam button (only shown for emotion mode)
        self.webcam_btn = tk.Button(
            mode_frame,
            text="📹 Use Webcam",
            command=self.start_webcam,
            font=self.fonts['small'],
            bg=self.colors['success'],
            fg="#000000",
            cursor="hand2",
            height=1,
            relief=tk.FLAT,
            activebackground="#00cc70"
        )
        # Initially hidden
        self.webcam_btn.pack_forget()
        
        # Action buttons frame
        action_frame = tk.Frame(left_panel, bg=self.colors['bg_medium'])
        action_frame.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        # Detect button
        self.detect_btn = tk.Button(
            action_frame,
            text="🚀 RUN DETECTION",
            command=self.run_detection,
            font=self.fonts['button'],
            bg=self.colors['accent'],
            fg="#000000",
            cursor="hand2",
            height=2,
            relief=tk.FLAT,
            state=tk.DISABLED,
            activebackground="#00b8e6",
            borderwidth=0
        )
        self.detect_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Stop button
        self.stop_btn = tk.Button(
            action_frame,
            text="⏹ STOP",
            command=self.stop_detection,
            font=self.fonts['button'],
            bg="#ff4444",
            fg="#ffffff",
            cursor="hand2",
            height=2,
            relief=tk.FLAT,
            state=tk.DISABLED,
            activebackground="#cc0000",
            borderwidth=0
        )
        self.stop_btn.pack(fill=tk.X)
        
        # Status section
        status_frame = tk.LabelFrame(
            left_panel,
            text="📊 Status Log",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            padx=20,
            pady=15,
            relief=tk.FLAT,
            borderwidth=0
        )
        status_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        self.status_text = scrolledtext.ScrolledText(
            status_frame,
            font=self.fonts['mono'],
            bg=self.colors['bg_light'],
            fg=self.colors['success'],
            height=15,
            wrap=tk.WORD,
            relief=tk.FLAT
        )
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.log_status("✓ System Ready. Please upload an image.")
        
        # Right panel - Images and Results
        right_panel = tk.Frame(content_frame, bg=self.colors['bg_dark'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Top: Images (larger to fill frame)
        images_frame = tk.Frame(right_panel, bg=self.colors['bg_dark'], height=450)
        images_frame.pack(fill=tk.X, pady=(0, 15))
        images_frame.pack_propagate(False)
        
        # Input image
        input_frame = tk.LabelFrame(
            images_frame,
            text="📷 Input Image",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            borderwidth=0,
            padx=5,
            pady=5
        )
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        self.input_image_label = tk.Label(
            input_frame,
            text="No image loaded\n\n📁 Click 'Choose Image' to start",
            bg=self.colors['bg_light'],
            fg=self.colors['text_dim'],
            font=self.fonts['text']
        )
        self.input_image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Output image
        output_frame = tk.LabelFrame(
            images_frame,
            text="✅ Output Image",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            borderwidth=0,
            padx=5,
            pady=5
        )
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))
        
        self.output_image_label = tk.Label(
            output_frame,
            text="Detection results will appear here\n\n🚀 Run detection to see results",
            bg=self.colors['bg_light'],
            fg=self.colors['text_dim'],
            font=self.fonts['text']
        )
        self.output_image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bottom: Details (much larger)
        details_frame = tk.LabelFrame(
            right_panel,
            text="📋 Detection Details",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            borderwidth=0,
            padx=5,
            pady=5
        )
        details_frame.pack(fill=tk.BOTH, expand=True)
        
        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            font=('Consolas', 12),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            wrap=tk.WORD,
            relief=tk.FLAT
        )
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def on_mode_change(self):
        """Handle mode change - show/hide webcam button"""
        if self.mode_var.get() == "face":
            self.webcam_btn.pack(anchor=tk.W, pady=10, fill=tk.X)
        else:
            self.webcam_btn.pack_forget()
    
    def start_webcam(self):
        """Start webcam emotion detection"""
        self.log_status("Starting webcam emotion detection...")
        if self.webcam_detector is None:
            self.webcam_detector = WebcamEmotionDetector()
        
        # Run in separate thread
        thread = threading.Thread(target=self.webcam_detector.start_webcam)
        thread.daemon = True
        thread.start()
    
    def log_status(self, message):
        """Add message to status log"""
        self.status_text.insert(tk.END, f"► {message}\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.detect_btn.config(state=tk.NORMAL)
            self.log_status(f"Loaded: {os.path.basename(file_path)}")
            
            # Display input image
            self.display_image(file_path, self.input_image_label)
            
            # Clear previous results
            self.output_image_label.config(image='', text="Detection results will appear here\n\n🚀 Run detection to see results", bg=self.colors['bg_light'])
            self.details_text.delete(1.0, tk.END)
    
    def display_image(self, image_path, label, is_output=False):
        """Display image in label"""
        try:
            # Load image
            image = Image.open(image_path)

            # Different sizes for input (smaller) vs output (larger)
            if is_output:
                # Output image - larger to show detection results clearly
                max_w, max_h = 600, 500
            else:
                # Input image - smaller
                max_w, max_h = 400, 450

            # Resize to fit while preserving aspect ratio (no cropping)
            image = ImageOps.contain(image, (max_w, max_h), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            label.config(image=photo, text="", bg=self.colors['bg_dark'], anchor='center')
            label.image = photo  # Keep reference
            
        except Exception as e:
            self.log_status(f"Error displaying image: {e}")
    
    def run_detection(self):
        """Run detection in separate thread"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        
        # Reset stop flag
        self.stop_flag = False
        
        # Disable detect button, enable stop button
        self.detect_btn.config(state=tk.DISABLED, text="⏳ PROCESSING...", bg=self.colors['text_dim'])
        self.stop_btn.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        mode = self.mode_var.get()
        
        # Run in thread to avoid freezing GUI
        self.processing_thread = threading.Thread(target=self.process_detection, args=(mode,))
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_detection(self):
        """Stop the current detection process"""
        self.stop_flag = True
        self.log_status("⏹ Stopping detection...")
        self.stop_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.NORMAL, text="🚀 RUN DETECTION", bg=self.colors['accent'])
    
    def process_detection(self, mode):
        """Process the detection based on selected mode"""
        try:
            if self.stop_flag:
                self.log_status("✗ Detection stopped by user")
                return
            
            if mode == "object":
                self.log_status("Running Object Detection...")
                if not self.stop_flag:
                    self.detect_objects()
            elif mode == "face":
                self.log_status("Running Face & Emotion Detection...")
                if not self.stop_flag:
                    self.detect_faces_emotions()
            elif mode == "classification":
                self.log_status("Running Image Classification...")
                if not self.stop_flag:
                    self.classify_image()
            elif mode == "segmentation":
                self.log_status("Running Image Segmentation...")
                if not self.stop_flag:
                    self.segment_image()
            elif mode == "traffic":
                self.log_status("Running Traffic Sign Recognition...")
                if not self.stop_flag:
                    self.recognize_traffic_signs()
                
        except Exception as e:
            if not self.stop_flag:
                self.log_status(f"✗ Error: {str(e)}")
                messagebox.showerror("Error", f"Detection failed: {str(e)}")
        finally:
            self.detect_btn.config(state=tk.NORMAL, text="🚀 RUN DETECTION", bg=self.colors['accent'])
            self.stop_btn.config(state=tk.DISABLED)
    
    def detect_objects(self):
        """Perform object detection"""
        try:
            # Load YOLO model
            if self.yolo_model is None:
                self.log_status("Loading YOLOv8 model...")
                self.yolo_model = YOLO('yolov8n.pt')
                self.log_status("✓ Model loaded")
            
            # Read image
            image = cv2.imread(self.current_image_path)
            
            # Run detection with lower confidence threshold for more detections
            self.log_status("Analyzing image...")
            results = self.yolo_model(image, conf=0.15, verbose=False)
            
            # Get annotated image
            annotated_image = results[0].plot()
            
            # Save to objects folder
            filename = os.path.basename(self.current_image_path)
            output_path = self.object_detection_output / f"detected_{filename}"
            cv2.imwrite(str(output_path), annotated_image)
            self.current_output_path = str(output_path)
            
            # Display output
            self.display_image(str(output_path), self.output_image_label, is_output=True)
            
            # Get detections
            detections = results[0].boxes
            self.log_status(f"✓ Found {len(detections)} objects")
            
            # Show details
            details = f"{'='*60}\n"
            details += f"   OBJECT DETECTION RESULTS\n"
            details += f"{'='*60}\n\n"
            details += f"Total objects detected: {len(detections)}\n\n"
            
            if len(detections) > 0:
                details += "▼ Detected Objects:\n"
                details += "─" * 60 + "\n\n"
                
                object_counts = {}
                for box in detections:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    # Count objects
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    
                    details += f"  🎯 {class_name.upper():15s} {confidence:6.1%} confidence\n"
                
                details += "\n" + "─" * 60 + "\n"
                details += "📊 Summary:\n\n"
                for obj, count in sorted(object_counts.items()):
                    details += f"  • {obj.capitalize():15s} Count: {count}\n"
            else:
                details += "\n⚠ No objects detected.\n\n"
                details += "Tips:\n"
                details += "  • Use an image with common objects\n"
                details += "  • Ensure better lighting/clarity\n"
                details += "  • Objects must be in the 80 COCO classes\n"
            
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.details_text.insert(1.0, details)
            self.log_status("✓ Detection complete!")
            
        except Exception as e:
            raise Exception(f"Object detection error: {str(e)}")
    
    def detect_faces_emotions(self):
        """Perform face and emotion detection"""
        try:
            self.log_status("Analyzing faces and emotions...")
            self.log_status("(This may take a moment...)")
            
            # Read image
            image = cv2.imread(self.current_image_path)
            
            # Analyze with DeepFace (only emotion for speed)
            results = DeepFace.analyze(
                img_path=self.current_image_path,
                actions=['emotion'],  # Only emotion - much faster
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'  # Faster detector
            )
            
            # Handle single or multiple faces
            if not isinstance(results, list):
                results = [results]
            
            # Draw on image
            for result in results:
                region = result['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # Draw rectangle (thicker for better visibility)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Get dominant emotion
                dominant_emotion = result['dominant_emotion']
                emotion_score = result['emotion'][dominant_emotion]
                
                # Draw label with larger font
                label = f"{dominant_emotion.upper()} {emotion_score:.0f}%"
                
                # Calculate label size for better background
                font_scale = 1.0
                thickness = 3
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Draw background rectangle
                cv2.rectangle(image, (x, y-text_height-20), (x+text_width+10, y), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(image, label, (x+5, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            # Save to emotions folder
            filename = os.path.basename(self.current_image_path)
            output_path = self.face_detection_output / f"emotion_{filename}"
            cv2.imwrite(str(output_path), image)
            self.current_output_path = str(output_path)
            
            # Display output
            self.display_image(str(output_path), self.output_image_label, is_output=True)
            
            self.log_status(f"✓ Found {len(results)} face(s)")
            
            # Show details
            details = f"{'='*60}\n"
            details += f"   FACE & EMOTION DETECTION RESULTS\n"
            details += f"{'='*60}\n\n"
            details += f"Total faces detected: {len(results)}\n\n"
            
            if len(results) > 0:
                for idx, result in enumerate(results, 1):
                    details += f"\n▼ Face {idx}:\n"
                    details += "─" * 60 + "\n"
                    
                    # Emotion
                    emotions = result['emotion']
                    dominant = result['dominant_emotion']
                    details += f"\n  🎭 EMOTION: {dominant.upper()}\n"
                    details += f"     Confidence: {emotions[dominant]:.1f}%\n"
                    
                    # All emotions
                    details += f"\n  📊 All Emotions:\n\n"
                    for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                        bar = "█" * int(score / 3.5)
                        details += f"     {emotion.capitalize():12s} {bar:20s} {score:5.1f}%\n"
                    
                    details += "\n"
            else:
                details += "\n⚠ No faces detected.\n\n"
                details += "Tips:\n"
                details += "  • Use an image with visible faces\n"
                details += "  • Ensure better lighting/resolution\n"
                details += "  • Use front-facing faces\n"
            
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.details_text.insert(1.0, details)
            self.log_status("✓ Detection complete!")
            
        except Exception as e:
            raise Exception(f"Face detection error: {str(e)}")
    
    def classify_image(self):
        """Perform image classification"""
        try:
            # Load classifier
            if self.classifier is None:
                self.log_status("Loading ResNet50 classifier...")
                self.classifier = ImageClassifier()
                self.classifier.load_model()
                self.log_status("✓ Model loaded")
            
            self.log_status("Classifying image...")
            
            # Save to classification folder
            filename = os.path.basename(self.current_image_path)
            output_path = self.classification_output / f"classified_{filename}"
            
            # Classify and get results
            predictions = self.classifier.classify_image(self.current_image_path)
            
            # Process with labels
            self.classifier.process_image_with_labels(self.current_image_path, output_path, predictions)
            
            self.current_output_path = str(output_path)
            
            # Display output
            self.display_image(str(output_path), self.output_image_label, is_output=True)
            
            self.log_status(f"✓ Classification complete!")
            
            # Show details
            details = f"{'='*60}\n"
            details += f"   IMAGE CLASSIFICATION RESULTS\n"
            details += f"{'='*60}\n\n"
            details += f"Top 5 Predictions:\n\n"
            
            for idx, (class_id, class_name, confidence) in enumerate(predictions, 1):
                bar = "█" * int(confidence * 50)
                details += f"  {idx}. {class_name.upper()}\n"
                details += f"     {bar:30s} {confidence:.1%}\n\n"
            
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.details_text.insert(1.0, details)
            
        except Exception as e:
            raise Exception(f"Classification error: {str(e)}")
    
    def segment_image(self):
        """Perform image segmentation"""
        try:
            # Load segmenter
            if self.segmenter is None:
                self.log_status("Loading segmentation model...")
                self.segmenter = ImageSegmenter()
                self.log_status("✓ Model loaded")
            
            self.log_status("Segmenting image...")
            
            # Save to segmentation folder
            filename = os.path.basename(self.current_image_path)
            output_path = self.segmentation_output / f"segmented_{filename}"
            
            # Segment image
            num_segments = self.segmenter.process_image(self.current_image_path, output_path)
            
            self.current_output_path = str(output_path)
            
            # Display output
            self.display_image(str(output_path), self.output_image_label, is_output=True)
            
            self.log_status(f"✓ Found {num_segments} segments")
            
            # Show details
            details = f"{'='*60}\n"
            details += f"   IMAGE SEGMENTATION RESULTS\n"
            details += f"{'='*60}\n\n"
            details += f"Total segments detected: {num_segments}\n\n"
            
            details += "Segmentation technique: Color-based region detection\n"
            details += "Output: Side-by-side comparison with original\n\n"
            
            details += "Segmentation helps identify:\n"
            details += "  • Different regions in the image\n"
            details += "  • Object boundaries\n"
            details += "  • Image structure\n"
            
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.details_text.insert(1.0, details)
            
        except Exception as e:
            raise Exception(f"Segmentation error: {str(e)}")
    
    def recognize_traffic_signs(self):
        """Perform traffic sign recognition"""
        try:
            # Load recognizer
            if self.traffic_recognizer is None:
                self.log_status("Loading traffic sign recognizer...")
                self.traffic_recognizer = TrafficSignRecognizer()
                self.log_status("✓ Model loaded")
            
            self.log_status("Recognizing traffic signs...")
            
            # Save to traffic folder
            filename = os.path.basename(self.current_image_path)
            output_path = self.traffic_output / f"detected_{filename}"
            
            # Recognize signs
            num_signs, detections = self.traffic_recognizer.process_image(
                self.current_image_path, output_path
            )
            
            self.current_output_path = str(output_path)
            
            # Display output
            self.display_image(str(output_path), self.output_image_label, is_output=True)
            
            self.log_status(f"✓ Found {num_signs} traffic signs")
            
            # Show details
            details = f"{'='*60}\n"
            details += f"   TRAFFIC SIGN RECOGNITION RESULTS\n"
            details += f"{'='*60}\n\n"
            details += f"Total signs detected: {num_signs}\n\n"
            
            if num_signs > 0:
                details += "▼ Detected Signs:\n"
                details += "─" * 60 + "\n\n"
                
                for idx, det in enumerate(detections, 1):
                    details += f"  {idx}. {det['type']}\n"
                    details += f"     Confidence: {det['confidence']:.1%}\n"
                    details += f"     Method: {det['method']}\n\n"
            else:
                details += "⚠ No traffic signs detected.\n\n"
                details += "Tips:\n"
                details += "  • Use clear images of traffic signs\n"
                details += "  • Ensure good lighting and contrast\n"
                details += "  • Signs should be clearly visible\n"
            
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.details_text.insert(1.0, details)
            
        except Exception as e:
            raise Exception(f"Traffic sign recognition error: {str(e)}")

def main():
    root = tk.Tk()
    app = ComputerVisionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
