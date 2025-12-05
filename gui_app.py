"""
Computer Vision GUI Application
Supports Object Detection and Face/Emotion Detection
Upload images and get results with detailed analysis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from deepface import DeepFace
from pathlib import Path
import threading
import os

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
        self.current_image_path = None
        self.current_output_path = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_styles(self):
        """Setup color scheme and fonts"""
        # Modern, accessible dark palette
        self.colors = {
            'bg_dark': '#0b1b2b',       # deep navy
            'bg_medium': '#112936',     # slate blue
            'bg_light': '#183343',      # muted slate
            'accent': '#00b4d8',        # cyan accent
            'success': '#4cd964',       # bright green
            'warning': '#ffb86b',       # warm orange
            'text': '#e6f6fb',          # off-white
            'text_dim': '#9fb7c6'       # soft muted text
        }

        # Fonts tuned for clarity and cross-platform availability
        self.fonts = {
            'title': ('Segoe UI', 26, 'bold'),
            'heading': ('Segoe UI', 16, 'bold'),
            'button': ('Segoe UI', 13, 'bold'),
            'text': ('Segoe UI', 12),
            'small': ('Segoe UI', 11),
            'mono': ('Consolas', 11)
        }
    
    def setup_directories(self):
        """Create necessary directories"""
        base_dir = Path(__file__).parent
        
        # Separate folders for different detection types
        self.object_detection_input = base_dir / "input_images" / "objects"
        self.face_detection_input = base_dir / "input_images" / "faces"
        self.object_detection_output = base_dir / "output_images" / "objects"
        self.face_detection_output = base_dir / "output_images" / "emotions"
        
        for directory in [self.object_detection_input, self.face_detection_input,
                         self.object_detection_output, self.face_detection_output]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_gui(self):
        """Setup the GUI layout"""
        
        # Title
        title_frame = tk.Frame(self.root, bg=self.colors['bg_medium'], height=100)
        title_frame.pack(fill=tk.X, padx=15, pady=15)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🔍 Computer Vision Detection System",
            font=self.fonts['title'],
            bg=self.colors['bg_medium'],
            fg=self.colors['accent']
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            title_frame,
            text="Object Detection & Emotion Recognition",
            font=self.fonts['small'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dim']
        )
        subtitle_label.pack()
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        # Left panel - Controls
        left_panel = tk.Frame(content_frame, bg=self.colors['bg_medium'], width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Upload section
        upload_frame = tk.LabelFrame(
            left_panel,
            text="📁 Upload Image",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            padx=15,
            pady=15
        )
        upload_frame.pack(fill=tk.X, padx=15, pady=15)
        
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
            activebackground="#00cc70"
        )
        self.upload_btn.pack(fill=tk.X, pady=8)
        
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
            padx=15,
            pady=15
        )
        mode_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.mode_var = tk.StringVar(value="object")
        
        object_radio = tk.Radiobutton(
            mode_frame,
            text="🎯 Object Detection (80 classes)",
            variable=self.mode_var,
            value="object",
            font=self.fonts['text'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            selectcolor=self.colors['bg_light'],
            activebackground=self.colors['bg_medium'],
            activeforeground=self.colors['accent']
        )
        object_radio.pack(anchor=tk.W, pady=8)
        
        face_radio = tk.Radiobutton(
            mode_frame,
            text="😊 Face & Emotion Detection",
            variable=self.mode_var,
            value="face",
            font=self.fonts['text'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            selectcolor=self.colors['bg_light'],
            activebackground=self.colors['bg_medium'],
            activeforeground=self.colors['accent']
        )
        face_radio.pack(anchor=tk.W, pady=8)
        
        # Detect button
        self.detect_btn = tk.Button(
            left_panel,
            text="🚀 RUN DETECTION",
            command=self.run_detection,
            font=self.fonts['button'],
            bg=self.colors['accent'],
            fg="#000000",
            cursor="hand2",
            height=3,
            relief=tk.FLAT,
            state=tk.DISABLED,
            activebackground="#00b8e6"
        )
        self.detect_btn.pack(fill=tk.X, padx=15, pady=20)
        
        # Status section
        status_frame = tk.LabelFrame(
            left_panel,
            text="📊 Status Log",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            padx=15,
            pady=15
        )
        status_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
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
        
        # Top: Images (compact for more details space)
        images_frame = tk.Frame(right_panel, bg=self.colors['bg_dark'], height=420)
        images_frame.pack(fill=tk.X, pady=(0, 10))
        images_frame.pack_propagate(False)
        
        # Input image
        input_frame = tk.LabelFrame(
            images_frame,
            text="📷 Input Image",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text']
        )
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        self.input_image_label = tk.Label(
            input_frame,
            text="No image loaded\n\n📁 Click 'Choose Image' to start",
            bg=self.colors['bg_light'],
            fg=self.colors['text_dim'],
            font=self.fonts['text']
        )
        self.input_image_label.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Output image
        output_frame = tk.LabelFrame(
            images_frame,
            text="✅ Output Image",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text']
        )
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))
        
        self.output_image_label = tk.Label(
            output_frame,
            text="Detection results will appear here\n\n🚀 Run detection to see results",
            bg=self.colors['bg_light'],
            fg=self.colors['text_dim'],
            font=self.fonts['text']
        )
        self.output_image_label.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Bottom: Details (much larger)
        details_frame = tk.LabelFrame(
            right_panel,
            text="📋 Detection Details",
            font=self.fonts['heading'],
            bg=self.colors['bg_medium'],
            fg=self.colors['text']
        )
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            font=('Consolas', 12),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            wrap=tk.WORD,
            relief=tk.FLAT
        )
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
    
    def display_image(self, image_path, label):
        """Display image in label"""
        try:
            image = Image.open(image_path).convert("RGBA")

            # Target frame size (do not crop — letterbox to preserve full image)
            max_w, max_h = (550, 480)

            # Original image size
            orig_w, orig_h = image.size

            # Compute scale to fit while preserving aspect ratio
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            # Resize the image with high-quality resampling
            resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Create background (letterbox) using the frame background color
            bg_hex = self.colors.get('bg_light', '#FFFFFF')
            bg_rgb = tuple(int(bg_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            background = Image.new('RGBA', (max_w, max_h), bg_rgb + (255,))

            # Paste centered
            paste_x = (max_w - new_w) // 2
            paste_y = (max_h - new_h) // 2
            background.paste(resized, (paste_x, paste_y), resized)

            # Convert to RGB for Tkinter and display
            final_image = background.convert('RGB')
            photo = ImageTk.PhotoImage(final_image)

            label.config(image=photo, text="", bg=self.colors['bg_light'])
            label.image = photo  # Keep reference
            
        except Exception as e:
            self.log_status(f"Error displaying image: {e}")
    
    def run_detection(self):
        """Run detection in separate thread"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        
        # Disable button during processing
        self.detect_btn.config(state=tk.DISABLED, text="⏳ PROCESSING...", bg=self.colors['text_dim'])
        self.details_text.delete(1.0, tk.END)
        
        mode = self.mode_var.get()
        
        # Run in thread to avoid freezing GUI
        thread = threading.Thread(target=self.process_detection, args=(mode,))
        thread.daemon = True
        thread.start()
    
    def process_detection(self, mode):
        """Process the detection based on selected mode"""
        try:
            if mode == "object":
                self.log_status("Running Object Detection...")
                self.detect_objects()
            else:
                self.log_status("Running Face & Emotion Detection...")
                self.detect_faces_emotions()
                
        except Exception as e:
            self.log_status(f"✗ Error: {str(e)}")
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
        finally:
            self.detect_btn.config(state=tk.NORMAL, text="🚀 RUN DETECTION", bg=self.colors['accent'])
    
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
            self.display_image(str(output_path), self.output_image_label)
            
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
            self.display_image(str(output_path), self.output_image_label)
            
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

def main():
    root = tk.Tk()
    app = ComputerVisionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
