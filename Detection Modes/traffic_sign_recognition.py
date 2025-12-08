"""
Traffic Sign Recognition using YOLO and Color Detection
Detects and classifies common traffic signs in images
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class TrafficSignRecognizer:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = None
        self.model_path = model_path
        
        # Common traffic sign categories based on color and shape
        self.sign_types = {
            'red_circle': 'Prohibitory Sign',
            'red_triangle': 'Warning Sign',
            'blue_circle': 'Mandatory Sign',
            'yellow_diamond': 'Caution Sign',
            'red_octagon': 'Stop Sign',
            'green_rect': 'Guide Sign'
        }
        
        self.colors = {
            'red': ([0, 100, 100], [10, 255, 255]),      # Red HSV range 1
            'red2': ([170, 100, 100], [180, 255, 255]),  # Red HSV range 2
            'blue': ([100, 100, 100], [130, 255, 255]),  # Blue
            'yellow': ([20, 100, 100], [30, 255, 255]),  # Yellow
            'green': ([40, 100, 100], [80, 255, 255])    # Green
        }
    
    def load_model(self):
        """Load YOLO model for object detection"""
        if self.model is None:
            print(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print("✓ Model loaded successfully!")
    
    def detect_shape(self, contour):
        """Detect shape of contour (circle, triangle, rectangle, octagon)"""
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        
        # Classify by number of vertices
        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return 'square'
            return 'rectangle'
        elif vertices == 8:
            return 'octagon'
        elif vertices > 8:
            # Check circularity
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                return 'circle'
        
        return 'unknown'
    
    def detect_color(self, roi_hsv):
        """Detect dominant color in ROI"""
        detected_colors = []
        
        for color_name, (lower, upper) in self.colors.items():
            mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
            ratio = cv2.countNonZero(mask) / (roi_hsv.shape[0] * roi_hsv.shape[1])
            if ratio > 0.1:  # At least 10% of the region
                detected_colors.append((color_name.replace('2', ''), ratio))
        
        if detected_colors:
            detected_colors.sort(key=lambda x: x[1], reverse=True)
            return detected_colors[0][0]
        
        return 'unknown'
    
    def classify_sign(self, color, shape):
        """Classify traffic sign based on color and shape"""
        key = f"{color}_{shape}"
        
        # Map color-shape combinations to sign types
        if color == 'red' and shape == 'octagon':
            return 'Stop Sign'
        elif color == 'red' and shape == 'circle':
            return 'Prohibitory Sign (No Entry/Speed Limit)'
        elif color == 'red' and shape == 'triangle':
            return 'Warning Sign (Yield/Caution)'
        elif color == 'blue' and shape == 'circle':
            return 'Mandatory Sign (Direction/Parking)'
        elif color == 'yellow' and shape in ['triangle', 'diamond']:
            return 'Caution/Warning Sign'
        elif color == 'green' and shape == 'rectangle':
            return 'Guide/Information Sign'
        elif shape == 'triangle':
            return 'Warning Sign'
        else:
            return f'Traffic Sign ({color} {shape})'
    
    def detect_signs_color_shape(self, image):
        """Detect traffic signs using color and shape detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        signs = []
        
        # Find potential sign regions
        for color_name, (lower, upper) in self.colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Filter by aspect ratio (signs are usually square-ish)
                    if 0.5 <= aspect_ratio <= 2.0:
                        roi_hsv = hsv[y:y+h, x:x+w]
                        shape = self.detect_shape(contour)
                        dominant_color = self.detect_color(roi_hsv)
                        sign_type = self.classify_sign(dominant_color, shape)
                        
                        signs.append({
                            'bbox': (x, y, w, h),
                            'color': dominant_color,
                            'shape': shape,
                            'type': sign_type,
                            'confidence': 0.8
                        })
        
        return signs
    
    def process_image(self, image_path, output_path, confidence_threshold=0.25):
        """Process image and detect traffic signs"""
        self.load_model()
        
        # Load image
        image = cv2.imread(str(image_path))
        original = image.copy()
        
        # Method 1: YOLO detection (for general objects that might be signs)
        results = self.model(image, conf=confidence_threshold)
        yolo_detections = []
        
        # Get YOLO detections for relevant objects
        relevant_classes = ['stop sign', 'parking meter', 'traffic light', 'fire hydrant']
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                
                if class_name in relevant_classes or 'sign' in class_name.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_detections.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'type': class_name.upper(),
                        'confidence': confidence,
                        'method': 'YOLO'
                    })
        
        # Method 2: Color-shape based detection
        shape_detections = self.detect_signs_color_shape(image)
        
        # Combine detections
        all_detections = yolo_detections + [
            {**d, 'method': 'Shape'} for d in shape_detections
        ]
        
        # Draw detections
        for detection in all_detections:
            x, y, w, h = detection['bbox']
            sign_type = detection['type']
            confidence = detection['confidence']
            method = detection['method']
            
            # Color based on method
            color = (0, 255, 0) if method == 'YOLO' else (255, 165, 0)
            
            # Draw box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            label = f"{sign_type}"
            conf_label = f"{confidence:.0%}"
            
            # Smaller font
            font_scale = 0.5
            thickness = 1
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, 1)
            
            bg_width = max(label_w, conf_w) + 10
            bg_height = label_h + conf_h + 15
            cv2.rectangle(image, (x, y - bg_height), (x + bg_width, y), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x + 5, y - conf_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            cv2.putText(image, conf_label, (x + 5, y - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), 1)
        
        # Add legend
        legend_height = 70
        legend = np.zeros((legend_height, image.shape[1], 3), dtype=np.uint8)
        legend[:] = (40, 40, 40)
        
        cv2.putText(legend, f"Traffic Signs Detected: {len(all_detections)}", (20, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(legend, f"YOLO: {len(yolo_detections)} | Shape-based: {len(shape_detections)}", 
                   (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        # Combine image with legend
        result_image = np.vstack([image, legend])
        
        # Save result
        cv2.imwrite(str(output_path), result_image)
        
        return len(all_detections), all_detections

def main():
    """Test traffic sign recognition"""
    print("="*60)
    print("TRAFFIC SIGN RECOGNITION")
    print("="*60)
    
    recognizer = TrafficSignRecognizer()
    
    # Create directories
    base_dir = Path(__file__).parent
    input_dir = base_dir / "input_images" / "traffic_signs"
    output_dir = base_dir / "output_images" / "traffic_signs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"\n⚠ No images found in {input_dir}")
        print("Please add traffic sign images to detect.")
        return
    
    print(f"\nFound {len(image_files)} image(s) to process\n")
    
    # Process each image
    for image_path in image_files:
        print(f"Processing: {image_path.name}")
        output_path = output_dir / f"detected_{image_path.name}"
        
        num_signs, detections = recognizer.process_image(image_path, output_path)
        
        print(f"  Signs detected: {num_signs}")
        for det in detections:
            print(f"    - {det['type']} (conf: {det['confidence']:.2f}, method: {det['method']})")
        print(f"  Output saved: {output_path.name}")
        print("-" * 60)
    
    print("\n" + "="*60)
    print("DETECTION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
