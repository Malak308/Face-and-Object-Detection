"""
Computer Vision Application - Modern GUI with CustomTkinter
Supports 5 CV modes: Object Detection, Emotion Recognition, Face Recognition, Segmentation, Traffic Signs
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
from ultralytics import YOLO
from deepface import DeepFace
from pathlib import Path
import threading
import os
from datetime import datetime
import queue

# Import CV modules from src/detectors
import sys
# Add parent directory to path to import from detectors
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from detectors.image_segmenter import ImageSegmenter
from detectors.traffic_sign_detector import TrafficSignRecognizer
from detectors.webcam_emotion_detector import WebcamEmotionDetector
from detectors.face_recognizer import FaceRecognitionSystem


class ComputerVisionGUI:
    """
    Modern GUI for Computer Vision using CustomTkinter
    """
    
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("🔍 Computer Vision Detection System v2.0")
        self.root.geometry("1500x900")
        
        # Set minimum window size
        self.root.minsize(1400, 850)
        
        # Custom colors matching the network tool
        self.colors = {
            'primary': '#1e40af',      # Deep blue
            'secondary': '#3b82f6',    # Bright blue
            'success': '#10b981',      # Green
            'danger': '#ef4444',       # Red
            'warning': '#f59e0b',      # Orange
            'info': '#06b6d4',         # Cyan
            'dark': '#1f2937',         # Dark gray
            'darker': '#111827',       # Darker gray
            'light': '#f3f4f6',        # Light gray
            'accent': '#8b5cf6'        # Purple
        }
        
        # Initialize models
        self.yolo_model = None
        self.segmenter = None
        self.traffic_recognizer = None
        self.webcam_detector = None
        self.face_recognizer = None
        
        # State variables
        self.current_image_path = None
        self.current_output_path = None
        self.stop_flag = False
        self.processing_thread = None
        self.output_queue = queue.Queue()
        
        # Webcam variables
        self.webcam_active = False
        self.webcam_cap = None
        self.webcam_thread = None
        self.webcam_mode = "face"  # Default mode: face emotion
        
        # Setup directories
        self.setup_directories()
        
        # Build GUI
        self.build_gui()
        
        # Check initial mode to show/hide webcam button
        self.on_mode_change()
        
        # Start output processor
        self.process_output()
        
    def setup_directories(self):
        """Create necessary directories"""
        # Use src/input_images and src/output_images as default
        base_dir = Path(__file__).parent.parent
        
        # Input folders
        self.object_input = base_dir / "input_images" / "objects"
        self.face_input = base_dir / "input_images" / "faces"
        self.segmentation_input = base_dir / "input_images" / "segmentation"
        self.traffic_input = base_dir / "input_images" / "traffic_signs"
        
        # Output folders
        self.object_output = base_dir / "output_images" / "objects"
        self.face_output = base_dir / "output_images" / "emotions"
        self.segmentation_output = base_dir / "output_images" / "segmentation"
        self.traffic_output = base_dir / "output_images" / "traffic_signs"
        
        all_dirs = [
            self.object_input, self.face_input, 
            self.segmentation_input, self.traffic_input,
            self.object_output, self.face_output,
            self.segmentation_output, self.traffic_output
        ]
        
        for directory in all_dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def build_gui(self):
        """Build the complete GUI layout"""
        # Configure grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Left sidebar
        self.create_sidebar()
        
        # Main content area
        self.create_main_content()
    
    def create_sidebar(self):
        """Create left sidebar with controls"""
        # Sidebar container
        sidebar_container = ctk.CTkFrame(self.root, width=380, corner_radius=0, fg_color=self.colors['darker'])
        sidebar_container.grid(row=0, column=0, sticky="nsew")
        sidebar_container.grid_rowconfigure(0, weight=1)
        sidebar_container.grid_columnconfigure(0, weight=1)
        
        # Scrollable sidebar
        sidebar = ctk.CTkScrollableFrame(
            sidebar_container,
            width=360,
            corner_radius=0,
            fg_color=self.colors['darker'],
            scrollbar_button_color=self.colors['primary'],
            scrollbar_button_hover_color=self.colors['secondary']
        )
        sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        sidebar.grid_columnconfigure(0, weight=1)
        
        # Logo/Title
        title_frame = ctk.CTkFrame(sidebar, fg_color=self.colors['primary'], corner_radius=10)
        title_frame.grid(row=0, column=0, padx=15, pady=(15, 20), sticky="ew")
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🔍 Computer Vision\nDetection System",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="white"
        )
        title_label.pack(pady=15)
        
        version_label = ctk.CTkLabel(
            sidebar,
            text="Version 2.0 | Educational Purpose",
            font=ctk.CTkFont(size=10),
            text_color="gray60"
        )
        version_label.grid(row=1, column=0, padx=20, pady=(0, 10))
        
        # Upload Section
        upload_label = ctk.CTkLabel(
            sidebar,
            text="📁 Upload Image",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['info']
        )
        upload_label.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.upload_btn = ctk.CTkButton(
            sidebar,
            text="📤 Choose Image",
            command=self.upload_image,
            width=340,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=self.colors['success'],
            hover_color="#059669"
        )
        self.upload_btn.grid(row=3, column=0, padx=20, pady=5)
        
        self.file_label = ctk.CTkLabel(
            sidebar,
            text="No file selected",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
            wraplength=320
        )
        self.file_label.grid(row=4, column=0, padx=20, pady=5)
        
        # Detection Mode Section
        mode_label = ctk.CTkLabel(
            sidebar,
            text="🎯 Detection Mode",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['warning']
        )
        mode_label.grid(row=5, column=0, padx=20, pady=(20, 5), sticky="w")
        
        # Mode selection frame
        mode_frame = ctk.CTkFrame(sidebar, fg_color=self.colors['dark'], corner_radius=10, border_width=2, border_color=self.colors['secondary'])
        mode_frame.grid(row=6, column=0, padx=20, pady=5, sticky="ew")
        
        self.mode_var = ctk.StringVar(value="object")
        
        modes = [
            ("Object Detection", "object"),
            ("Face & Emotion", "face"),
            ("Face Recognition", "recognition"),
            ("Image Segmentation", "segmentation"),
            ("Traffic Sign Recognition", "traffic")
        ]
        
        for idx, (text, value) in enumerate(modes):
            radio = ctk.CTkRadioButton(
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                font=ctk.CTkFont(size=12),
                command=self.on_mode_change,
                fg_color=self.colors['primary'],
                hover_color=self.colors['secondary']
            )
            radio.grid(row=idx, column=0, padx=15, pady=5, sticky="w")
        
        # Store sidebar reference for webcam button
        self.sidebar = sidebar
        
        # Webcam button (initially not displayed)
        self.webcam_btn = ctk.CTkButton(
            sidebar,
            text="📹 Use Webcam",
            command=self.start_webcam,
            width=340,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=self.colors['accent'],
            hover_color="#7c3aed"
        )
        # Don't grid it yet - will appear when face mode is selected
        
        # Action Buttons
        action_label = ctk.CTkLabel(
            sidebar,
            text="⚡ Actions",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['accent']
        )
        action_label.grid(row=8, column=0, padx=20, pady=(20, 5), sticky="w")
        
        self.detect_btn = ctk.CTkButton(
            sidebar,
            text="🚀 RUN DETECTION",
            command=self.run_detection,
            width=340,
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=self.colors['primary'],
            hover_color=self.colors['secondary'],
            state="disabled"
        )
        self.detect_btn.grid(row=9, column=0, padx=20, pady=8)
        
        self.stop_btn = ctk.CTkButton(
            sidebar,
            text="⛔ STOP",
            command=self.stop_detection,
            width=340,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=self.colors['danger'],
            hover_color="#dc2626",
            state="disabled"
        )
        self.stop_btn.grid(row=10, column=0, padx=20, pady=5)
        
        clear_btn = ctk.CTkButton(
            sidebar,
            text="🗑️ Clear All",
            command=self.clear_all,
            width=340,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=12),
            fg_color=self.colors['dark'],
            hover_color="#374151",
            border_width=1,
            border_color="gray50"
        )
        clear_btn.grid(row=11, column=0, padx=20, pady=5)
        
        # Status Log Section
        log_label = ctk.CTkLabel(
            sidebar,
            text="📊 Status Log",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['info']
        )
        log_label.grid(row=12, column=0, padx=20, pady=(20, 5), sticky="w")
        
        self.status_text = ctk.CTkTextbox(
            sidebar,
            width=340,
            height=200,
            corner_radius=10,
            border_width=2,
            border_color=self.colors['secondary'],
            fg_color="#0f172a",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color="#a5f3fc"
        )
        self.status_text.grid(row=13, column=0, padx=20, pady=(5, 20))
        self.log_status("✅ System Ready")
        self.log_status("Upload an image to start detection")
    
    def create_main_content(self):
        """Create main content area"""
        # Main frame
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Status bar at top
        status_frame = ctk.CTkFrame(main_frame, fg_color=self.colors['dark'], corner_radius=10, height=60)
        status_frame.grid(row=0, column=0, padx=10, pady=(10, 15), sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="✅ Ready to detect",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
            text_color=self.colors['success']
        )
        self.status_label.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        self.status_icon = ctk.CTkLabel(
            status_frame,
            text="●",
            font=ctk.CTkFont(size=24),
            text_color=self.colors['success']
        )
        self.status_icon.grid(row=0, column=1, padx=20, pady=15, sticky="e")
        
        # Tabview for results
        self.tabview = ctk.CTkTabview(
            main_frame,
            corner_radius=15,
            border_width=2,
            border_color=self.colors['primary'],
            fg_color=self.colors['dark'],
            segmented_button_fg_color=self.colors['darker'],
            segmented_button_selected_color=self.colors['primary'],
            segmented_button_selected_hover_color=self.colors['secondary']
        )
        self.tabview.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create tabs
        self.tabview.add("📷 Images")
        self.tabview.add("📋 Details")
        
        # Setup tabs
        self.setup_images_tab()
        self.setup_details_tab()
    
    def setup_images_tab(self):
        """Setup images display tab"""
        tab = self.tabview.tab("📷 Images")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        # Input image frame
        input_frame = ctk.CTkFrame(tab, fg_color=self.colors['darker'], corner_radius=10, border_width=2, border_color=self.colors['info'])
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        input_label = ctk.CTkLabel(
            input_frame,
            text="📷 Input Image",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['info']
        )
        input_label.pack(pady=10)
        
        self.input_image_label = ctk.CTkLabel(
            input_frame,
            text="No image loaded\n\n📁 Click 'Choose Image' to start",
            font=ctk.CTkFont(size=12),
            text_color="gray60"
        )
        self.input_image_label.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Output image frame
        output_frame = ctk.CTkFrame(tab, fg_color=self.colors['darker'], corner_radius=10, border_width=2, border_color=self.colors['success'])
        output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        output_label = ctk.CTkLabel(
            output_frame,
            text="✅ Output Image",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['success']
        )
        output_label.pack(pady=10)
        
        self.output_image_label = ctk.CTkLabel(
            output_frame,
            text="Detection results will appear here\n\n🚀 Run detection to see results",
            font=ctk.CTkFont(size=12),
            text_color="gray60"
        )
        self.output_image_label.pack(fill="both", expand=True, padx=15, pady=15)
    
    def setup_details_tab(self):
        """Setup detection details tab"""
        tab = self.tabview.tab("📋 Details")
        
        details_header = ctk.CTkLabel(
            tab,
            text="📊 Detection Results & Analysis",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['warning']
        )
        details_header.pack(pady=15)
        
        self.details_text = ctk.CTkTextbox(
            tab,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=10,
            border_width=2,
            border_color=self.colors['accent'],
            fg_color=self.colors['darker']
        )
        self.details_text.pack(fill="both", expand=True, padx=15, pady=10)
    
    def on_mode_change(self):
        """Handle mode change - show/hide webcam button and stop webcam if active"""
        if self.mode_var.get() in ["face", "recognition"]:
            # Show webcam button below mode selection (row 7, between mode frame and actions)
            self.webcam_btn.grid(row=7, column=0, padx=20, pady=(10, 5), sticky="ew")
        else:
            # Hide webcam button for other modes
            self.webcam_btn.grid_forget()
            # Stop webcam if it's currently active
            if self.webcam_active:
                self.stop_webcam()
    
    def start_webcam(self):
        """Start webcam detection (emotion or recognition based on mode)"""
        if self.webcam_active:
            self.stop_webcam()
            return
        
        # Determine mode
        current_mode = self.mode_var.get()
        mode_name = "emotion detection" if current_mode == "face" else "face recognition"
        
        self.log_status(f"📹 Starting webcam {mode_name}...")
        self.webcam_active = True
        self.webcam_mode = current_mode  # Store current mode
        self.webcam_btn.configure(text="⏹ Stop Webcam", fg_color=self.colors['danger'])
        
        # Switch to Images tab to show webcam feed
        self.tabview.set("📷 Images")
        
        # Initialize webcam
        self.webcam_cap = cv2.VideoCapture(0)
        if not self.webcam_cap.isOpened():
            self.log_status("❌ Error: Could not open webcam")
            self.webcam_active = False
            self.webcam_btn.configure(text="📹 Use Webcam", fg_color=self.colors['accent'])
            return
        
        # Set resolution
        self.webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.log_status(f"✅ Webcam started - {mode_name} active!")
        
        # Initialize frame counters
        self.frame_count = 0
        self.analyze_interval = 10
        self.last_analysis = None
        
        # Start webcam loop
        self.webcam_loop()
    
    def stop_webcam(self):
        """Stop webcam detection"""
        self.log_status("⏹ Stopping webcam...")
        self.webcam_active = False
        
        if self.webcam_cap:
            self.webcam_cap.release()
            self.webcam_cap = None
        
        self.webcam_btn.configure(text="📹 Use Webcam", fg_color=self.colors['accent'])
        self.log_status("✅ Webcam stopped")
    
    def webcam_loop(self):
        """Main webcam processing loop - initialize based on mode"""
        if self.webcam_mode == "face":
            # Emotion detection mode
            if self.webcam_detector is None:
                self.webcam_detector = WebcamEmotionDetector()
        elif self.webcam_mode == "recognition":
            # Face recognition mode
            if self.face_recognizer is None:
                self.face_recognizer = FaceRecognitionSystem()
                self.face_recognizer.load_database()
        
        self.frame_count = 0
        self.analyze_interval = 10
        self.last_analysis = None
        
        # Start the update cycle in main thread
        self.root.after(10, self.update_webcam_frame)
    
    def update_webcam_frame(self):
        """Update webcam frame - runs in main thread"""
        if not self.webcam_active or not self.webcam_cap:
            return
        
        ret, frame = self.webcam_cap.read()
        
        if not ret:
            self.log_status("❌ Failed to capture frame")
            self.stop_webcam()
            return
        
        # Analyze periodically based on mode
        if self.frame_count % self.analyze_interval == 0:
            try:
                if self.webcam_mode == "face":
                    # Emotion detection
                    self.last_analysis = self.webcam_detector.analyze_frame(frame)
                elif self.webcam_mode == "recognition":
                    # Face recognition
                    self.last_analysis = self.analyze_frame_for_recognition(frame)
            except Exception as e:
                self.log_status(f"⚠ Analysis error: {str(e)}")
                self.last_analysis = None
        
        # Draw info based on mode
        if self.last_analysis:
            if self.webcam_mode == "face":
                # Draw emotion info
                frame = self.webcam_detector.draw_emotion_info(frame, self.last_analysis)
            elif self.webcam_mode == "recognition":
                # Draw recognition info
                frame = self.draw_recognition_info(frame, self.last_analysis)
        
        # Add overlay with mode indicator
        mode_text = "LIVE EMOTION DETECTION" if self.webcam_mode == "face" else "LIVE FACE RECOGNITION"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, mode_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Convert to PIL Image for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize to fit display
        display_size = (640, 480)
        pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Update GUI
        photo = ImageTk.PhotoImage(image=pil_image)
        self.output_image_label.configure(image=photo, text="")
        self.output_image_label.image = photo
        
        self.frame_count += 1
        
        # Schedule next frame update (30 FPS = ~33ms)
        if self.webcam_active:
            self.root.after(33, self.update_webcam_frame)
    
    def analyze_frame_for_recognition(self, frame):
        """Analyze a webcam frame for face recognition"""
        try:
            # Detect faces
            results = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False,
                align=True
            )
            
            if not results:
                return None
            
            # Process each face
            face_data = []
            for idx, result in enumerate(results):
                facial_area = result['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                # Extract face
                face_img = frame[max(0, y):min(frame.shape[0], y+h), 
                                max(0, x):min(frame.shape[1], x+w)]
                
                if face_img.size == 0:
                    continue
                
                # Save temp face
                temp_path = f"temp_webcam_face_{idx}.jpg"
                cv2.imwrite(temp_path, face_img)
                
                # Recognize
                person_name = "Unknown"
                confidence = 0
                
                if self.face_recognizer.known_faces:
                    recognized_name, conf = self.face_recognizer.recognize_face(temp_path)
                    if recognized_name:
                        person_name = recognized_name.replace('_', ' ')
                        confidence = conf
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                face_data.append({
                    'region': facial_area,
                    'name': person_name,
                    'confidence': confidence
                })
            
            return face_data if face_data else None
            
        except Exception as e:
            return None
    
    def draw_recognition_info(self, frame, face_data):
        """Draw recognition information on webcam frame"""
        if not face_data:
            return frame
        
        for face in face_data:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            name = face['name']
            confidence = face['confidence']
            
            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Prepare label
            if name != "Unknown":
                label = f"{name} ({confidence:.0f}%)"
            else:
                label = "UNKNOWN"
            
            # Draw label background
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.9
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Background rectangle
            cv2.rectangle(frame, (x, y - text_h - 15), (x + text_w + 10, y), color, -1)
            
            # Text
            cv2.putText(frame, label, (x + 5, y - 8),
                       font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def log_status(self, message):
        """Add message to status log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_queue.put(f"[{timestamp}] {message}")
    
    def process_output(self):
        """Process output queue and update console"""
        try:
            while True:
                message = self.output_queue.get_nowait()
                self.status_text.insert("end", message + "\n")
                self.status_text.see("end")
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_output)
    
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
            self.file_label.configure(text=os.path.basename(file_path))
            self.detect_btn.configure(state="normal")
            self.log_status(f"✓ Loaded: {os.path.basename(file_path)}")
            
            # Display input image
            self.display_image(file_path, self.input_image_label)
            
            # Clear previous results
            self.output_image_label.configure(image='', text="Detection results will appear here\n\n🚀 Run detection to see results")
            self.details_text.delete("1.0", "end")
    
    def display_image(self, image_path, label, is_output=False):
        """Display image in label with enhanced canvas size"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Enhanced larger sizes for better visibility
            if is_output:
                max_w, max_h = 700, 600  # Increased from 650x550
            else:
                max_w, max_h = 700, 600  # Increased from 650x550
            
            # Resize while preserving aspect ratio with high-quality resampling
            image = ImageOps.contain(image, (max_w, max_h), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo, text="")
            label.image = photo  # Keep reference
            
        except Exception as e:
            self.log_status(f"✗ Error displaying image: {e}")
    
    def run_detection(self):
        """Run detection in separate thread"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        
        # Reset stop flag
        self.stop_flag = False
        
        # Update UI
        self.detect_btn.configure(state="disabled", text="⏳ PROCESSING...")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="🔍 Detection in progress...", text_color=self.colors['warning'])
        self.status_icon.configure(text_color=self.colors['warning'])
        self.details_text.delete("1.0", "end")
        
        mode = self.mode_var.get()
        
        # Run in thread
        self.processing_thread = threading.Thread(target=self.process_detection, args=(mode,))
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_detection(self):
        """Stop the current detection process"""
        # Stop webcam if active
        if self.webcam_active:
            self.stop_webcam()
        
        self.stop_flag = True
        self.log_status("⛔ Stopping detection...")
        self.stop_btn.configure(state="disabled")
        self.detect_btn.configure(state="normal", text="🚀 RUN DETECTION")
    
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
            elif mode == "recognition":
                self.log_status("Running Face Recognition...")
                if not self.stop_flag:
                    self.recognize_faces()
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
            self.root.after(0, self.finish_detection)
    
    def finish_detection(self):
        """Cleanup after detection finishes"""
        self.detect_btn.configure(state="normal", text="🚀 RUN DETECTION")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="✅ Detection complete - Ready for next scan", text_color=self.colors['success'])
        self.status_icon.configure(text_color=self.colors['success'])
    
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
            
            # Run detection
            self.log_status("Analyzing image...")
            results = self.yolo_model(image, conf=0.15, verbose=False)
            
            # Get annotated image
            annotated_image = results[0].plot()
            
            # Save
            filename = os.path.basename(self.current_image_path)
            output_path = self.object_output / f"detected_{filename}"
            cv2.imwrite(str(output_path), annotated_image)
            self.current_output_path = str(output_path)
            
            # Display output
            self.root.after(0, lambda: self.display_image(str(output_path), self.output_image_label, is_output=True))
            
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
                    
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    details += f"  🎯 {class_name.upper():15s} {confidence:6.1%} confidence\n"
                
                details += "\n" + "─" * 60 + "\n"
                details += "📊 Summary:\n\n"
                for obj, count in sorted(object_counts.items()):
                    details += f"  • {obj.capitalize():15s} Count: {count}\n"
            else:
                details += "\n⚠ No objects detected.\n"
            
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.root.after(0, lambda: self.details_text.insert("1.0", details))
            self.log_status("✓ Detection complete!")
            
        except Exception as e:
            raise Exception(f"Object detection error: {str(e)}")
    
    def detect_faces_emotions(self):
        """Perform face and emotion detection"""
        try:
            self.log_status("Analyzing faces and emotions...")
            
            # Read image
            image = cv2.imread(self.current_image_path)
            
            # Analyze with DeepFace
            results = DeepFace.analyze(
                img_path=self.current_image_path,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'
            )
            
            if not isinstance(results, list):
                results = [results]
            
            # Draw on image
            for result in results:
                region = result['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                dominant_emotion = result['dominant_emotion']
                emotion_score = result['emotion'][dominant_emotion]
                
                label = f"{dominant_emotion.upper()} {emotion_score:.0f}%"
                
                font_scale = 1.0
                thickness = 3
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                cv2.rectangle(image, (x, y-text_height-20), (x+text_width+10, y), (0, 255, 0), -1)
                cv2.putText(image, label, (x+5, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            # Save
            filename = os.path.basename(self.current_image_path)
            output_path = self.face_output / f"emotion_{filename}"
            cv2.imwrite(str(output_path), image)
            self.current_output_path = str(output_path)
            
            # Display output
            self.root.after(0, lambda: self.display_image(str(output_path), self.output_image_label, is_output=True))
            
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
                    
                    emotions = result['emotion']
                    dominant = result['dominant_emotion']
                    details += f"\n  🎭 EMOTION: {dominant.upper()}\n"
                    details += f"     Confidence: {emotions[dominant]:.1f}%\n"
                    
                    details += f"\n  📊 All Emotions:\n\n"
                    for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                        bar = "█" * int(score / 3.5)
                        details += f"     {emotion.capitalize():12s} {bar:20s} {score:5.1f}%\n"
                    
                    details += "\n"
            else:
                details += "\n⚠ No faces detected.\n"
            
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.root.after(0, lambda: self.details_text.insert("1.0", details))
            self.log_status("✓ Detection complete!")
            
        except Exception as e:
            raise Exception(f"Face detection error: {str(e)}")
    
    def recognize_faces(self):
        """Perform face recognition - identify people only (no emotion/age)"""
        try:
            # Initialize face recognizer if needed
            if self.face_recognizer is None:
                self.log_status("Initializing face recognition system...")
                self.face_recognizer = FaceRecognitionSystem()
                self.face_recognizer.load_database()
                self.log_status("✓ System ready")
            
            self.log_status("Identifying faces...")
            
            filename = os.path.basename(self.current_image_path)
            
            # Create recognized_faces output folder
            recognized_output = Path(__file__).parent.parent / "output_images" / "recognized_faces"
            recognized_output.mkdir(parents=True, exist_ok=True)
            output_path = recognized_output / f"identified_{filename}"
            
            # Process image with recognition (identity only)
            self.face_recognizer.process_image(self.current_image_path, output_path)
            
            self.current_output_path = str(output_path)
            
            # Display output
            self.root.after(0, lambda: self.display_image(str(output_path), self.output_image_label, is_output=True))
            
            # Count faces detected
            try:
                results = DeepFace.extract_faces(
                    img_path=self.current_image_path,
                    detector_backend='opencv',
                    enforce_detection=False
                )
                num_faces = len(results)
            except:
                num_faces = 0
            
            self.log_status(f"✓ Processed {num_faces} face(s)")
            
            # Show details
            details = f"{'='*60}\n"
            details += f"   FACE IDENTIFICATION RESULTS\n"
            details += f"{'='*60}\n\n"
            details += f"Mode: Identity Recognition (no emotion/age analysis)\n\n"
            details += f"Total faces detected: {num_faces}\n\n"
            
            if self.face_recognizer.known_faces:
                details += f"📁 Database Status:\n"
                details += f"   ✓ {len(self.face_recognizer.known_faces)} person(s) loaded\n\n"
                
                details += "👥 Known People:\n"
                for person_name, images in self.face_recognizer.known_faces.items():
                    details += f"   • {person_name.replace('_', ' ')} ({len(images)} reference photo(s))\n"
                
                details += f"\n{'─' * 60}\n"
                details += "\n🔍 Recognition Results:\n"
                details += "   • 🟢 Green box = Person identified\n"
                details += "   • 🔴 Red box = Unknown person\n"
                details += "   • See image for names and confidence scores\n"
            else:
                details += "⚠️  Database Empty\n\n"
                details += "All faces will be labeled as 'Unknown'\n\n"
                details += "📝 To add people to database:\n"
                details += "   1. Go to: src/input_images/face_database/\n"
                details += "   2. Create folder: Person_Name/\n"
                details += "   3. Add 1-3 clear photos of the person\n"
                details += "   4. Restart app to load database\n"
            
            details += f"\n{'─' * 60}\n"
            details += "\n💡 Note: For emotion/age/gender analysis,\n"
            details += "   use 'Face & Emotion' mode instead.\n"
            
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.root.after(0, lambda: self.details_text.insert("1.0", details))
            self.log_status("✓ Identification complete!")
            
        except Exception as e:
            raise Exception(f"Face recognition error: {str(e)}")
    
    def segment_image(self):
        """Perform image segmentation"""
        try:
            if self.segmenter is None:
                self.log_status("Loading segmentation model...")
                self.segmenter = ImageSegmenter()
                self.log_status("✓ Model loaded")
            
            self.log_status("Segmenting image...")
            
            filename = os.path.basename(self.current_image_path)
            output_path = self.segmentation_output / f"segmented_{filename}"
            
            num_segments = self.segmenter.process_image(self.current_image_path, output_path)
            
            self.current_output_path = str(output_path)
            
            self.root.after(0, lambda: self.display_image(str(output_path), self.output_image_label, is_output=True))
            
            self.log_status(f"✓ Found {num_segments} segments")
            
            details = f"{'='*60}\n"
            details += f"   IMAGE SEGMENTATION RESULTS\n"
            details += f"{'='*60}\n\n"
            details += f"Total segments detected: {num_segments}\n\n"
            details += "Segmentation technique: Color-based region detection\n"
            details += "Output: Side-by-side comparison with original\n"
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.root.after(0, lambda: self.details_text.insert("1.0", details))
            
        except Exception as e:
            raise Exception(f"Segmentation error: {str(e)}")
    
    def recognize_traffic_signs(self):
        """Perform traffic sign recognition"""
        try:
            if self.traffic_recognizer is None:
                self.log_status("Loading traffic sign recognizer...")
                self.traffic_recognizer = TrafficSignRecognizer()
                self.log_status("✓ Model loaded")
            
            self.log_status("Recognizing traffic signs...")
            
            filename = os.path.basename(self.current_image_path)
            output_path = self.traffic_output / f"detected_{filename}"
            
            num_signs, detections = self.traffic_recognizer.process_image(
                self.current_image_path, output_path
            )
            
            self.current_output_path = str(output_path)
            
            self.root.after(0, lambda: self.display_image(str(output_path), self.output_image_label, is_output=True))
            
            self.log_status(f"✓ Found {num_signs} traffic signs")
            
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
                details += "⚠ No traffic signs detected.\n"
            
            details += f"\n{'='*60}\n"
            details += f"✓ Output saved: {output_path.name}\n"
            
            self.root.after(0, lambda: self.details_text.insert("1.0", details))
            
        except Exception as e:
            raise Exception(f"Traffic sign recognition error: {str(e)}")
    
    def clear_all(self):
        """Clear all images and results"""
        self.current_image_path = None
        self.current_output_path = None
        
        self.file_label.configure(text="No file selected")
        self.input_image_label.configure(image='', text="No image loaded\n\n📁 Click 'Choose Image' to start")
        self.output_image_label.configure(image='', text="Detection results will appear here\n\n🚀 Run detection to see results")
        self.details_text.delete("1.0", "end")
        self.status_text.delete("1.0", "end")
        
        self.detect_btn.configure(state="disabled")
        self.status_label.configure(text="✅ Ready to detect", text_color=self.colors['success'])
        self.status_icon.configure(text_color=self.colors['success'])
        
        self.log_status("🗑️ Cleared all data")
        self.log_status("Ready for new detection")
    
    def run(self):
        """Start the GUI main loop"""
        self.log_status("🚀 Computer Vision System initialized")
        self.log_status("Select a mode and upload an image")
        
        # Bring window to front
        self.root.lift()
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))
        
        self.root.mainloop()


def main():
    """Main entry point"""
    app = ComputerVisionGUI()
    app.run()


if __name__ == "__main__":
    main()
