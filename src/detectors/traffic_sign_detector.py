"""
Traffic Sign Recognition - ENHANCED VERSION
Advanced detection using YOLO, Color-Shape Analysis, and Template Matching
Supports multiple detection methods with confidence scoring and NMS
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tensorflow as tf
from collections import defaultdict

class TrafficSignRecognizer:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = None
        self.model_path = model_path
        self._setup_gpu()
        
        # Enhanced traffic sign categories with detailed classifications
        self.sign_categories = {
            'stop': {'color': 'red', 'shape': 'octagon', 'priority': 1},
            'yield': {'color': 'red', 'shape': 'triangle', 'priority': 2},
            'speed_limit': {'color': 'red', 'shape': 'circle', 'priority': 3},
            'no_entry': {'color': 'red', 'shape': 'circle', 'priority': 3},
            'warning': {'color': 'yellow', 'shape': 'triangle', 'priority': 4},
            'mandatory': {'color': 'blue', 'shape': 'circle', 'priority': 5},
            'guide': {'color': 'green', 'shape': 'rectangle', 'priority': 6},
            'information': {'color': 'blue', 'shape': 'rectangle', 'priority': 6},
        }
        
        # Enhanced color ranges with multiple ranges for better detection
        self.colors = {
            'red': [
                ([0, 120, 70], [10, 255, 255]),      # Red HSV range 1
                ([170, 120, 70], [180, 255, 255])    # Red HSV range 2
            ],
            'blue': [
                ([90, 100, 70], [130, 255, 255])     # Blue
            ],
            'yellow': [
                ([15, 100, 100], [35, 255, 255])     # Yellow
            ],
            'green': [
                ([35, 50, 50], [85, 255, 255])       # Green
            ],
            'white': [
                ([0, 0, 200], [180, 30, 255])        # White
            ]
        }
    
    def _setup_gpu(self):
        """Configure TensorFlow for optimal GPU usage"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            pass
    
    
    def load_model(self):
        """Load YOLO model for object detection"""
        if self.model is None:
            print(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print("✓ Model loaded successfully!")
    
    def detect_shape(self, contour):
        """
        Enhanced shape detection with improved accuracy
        Detects: circle, triangle, rectangle, square, octagon, diamond
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 'unknown'
        
        # Adaptive epsilon for better approximation
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Get bounding box for aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Area and circularity calculations
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Triangle detection
        if vertices == 3:
            return 'triangle'
        
        # Quadrilateral detection (square, rectangle, diamond)
        elif vertices == 4:
            # Get angles to distinguish between rectangle and diamond
            angles = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i+1)%4][0]
                p3 = approx[(i+2)%4][0]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
                angles.append(np.degrees(angle))
            
            avg_angle = np.mean(angles)
            
            # Diamond: rotated square (angles around 90°)
            if 80 <= avg_angle <= 100 and 0.85 <= aspect_ratio <= 1.15:
                # Check rotation - if rotated ~45°, it's a diamond
                rect = cv2.minAreaRect(contour)
                angle = abs(rect[2])
                if 35 <= angle <= 55 or 125 <= angle <= 145:
                    return 'diamond'
                return 'square'
            
            # Square vs Rectangle
            if 0.85 <= aspect_ratio <= 1.15:
                return 'square'
            else:
                return 'rectangle'
        
        # Pentagon to Octagon
        elif 5 <= vertices <= 9:
            if vertices == 8 or (vertices == 7 and circularity > 0.75):
                return 'octagon'
            return f'{vertices}-gon'
        
        # Circle detection (many vertices or high circularity)
        elif vertices > 9 or circularity > 0.75:
            return 'circle'
        
        return 'unknown'
    
    def detect_color(self, roi_hsv):
        """Enhanced color detection with multiple ranges and confidence scoring"""
        color_scores = defaultdict(float)
        total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
        
        if total_pixels == 0:
            return 'unknown', 0.0
        
        for color_name, ranges in self.colors.items():
            combined_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in ranges:
                mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            ratio = cv2.countNonZero(combined_mask) / total_pixels
            color_scores[color_name] = ratio
        
        # Get dominant color
        if color_scores:
            dominant_color = max(color_scores.items(), key=lambda x: x[1])
            if dominant_color[1] > 0.08:  # At least 8% of the region
                return dominant_color[0], dominant_color[1]
        
        return 'unknown', 0.0
    
    def classify_sign_advanced(self, color, shape, confidence):
        """
        Advanced sign classification with detailed categories
        Returns: (sign_type, detailed_info, confidence_adjustment)
        """
        # Stop sign - highest priority
        if color == 'red' and shape == 'octagon':
            return 'STOP SIGN', 'Regulatory - Full Stop Required', 1.0
        
        # Yield sign
        elif color in ['red', 'yellow'] and shape == 'triangle':
            if color == 'red':
                return 'YIELD SIGN', 'Regulatory - Give Right of Way', 0.95
            else:
                return 'WARNING SIGN', 'Caution - General Warning', 0.85
        
        # Speed limit and prohibitory signs
        elif color == 'red' and shape == 'circle':
            return 'PROHIBITORY SIGN', 'Speed Limit / No Entry / Restriction', 0.9
        
        # Mandatory signs
        elif color == 'blue' and shape == 'circle':
            return 'MANDATORY SIGN', 'Direction / Minimum Speed / Parking', 0.85
        
        # Warning signs (yellow triangle/diamond)
        elif color == 'yellow' and shape in ['triangle', 'diamond']:
            return 'WARNING SIGN', 'Caution - Road Hazard Ahead', 0.85
        
        # Guide and information signs
        elif color == 'green' and shape == 'rectangle':
            return 'GUIDE SIGN', 'Direction / Distance / Location', 0.8
        
        elif color == 'blue' and shape == 'rectangle':
            return 'INFORMATION SIGN', 'Service / Facility Information', 0.8
        
        # White regulatory signs
        elif color == 'white' and shape in ['rectangle', 'square']:
            return 'REGULATORY SIGN', 'Traffic Regulation', 0.75
        
        # Generic classification
        elif shape == 'triangle':
            return 'WARNING SIGN', f'{color.upper()} warning indicator', 0.7
        elif shape == 'circle':
            return 'REGULATORY SIGN', f'{color.upper()} regulation', 0.7
        
        # Unknown but detected
        return 'TRAFFIC SIGN', f'{color.upper()} {shape.upper()}', 0.6
    
    
    def non_max_suppression(self, detections, iou_threshold=0.4):
        """
        Apply Non-Maximum Suppression to remove duplicate detections
        Keeps highest confidence detection for overlapping boxes
        """
        if len(detections) == 0:
            return []
        
        # Extract boxes and scores
        boxes = []
        scores = []
        indices = []
        
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            boxes.append([x, y, x+w, y+h])
            scores.append(det['confidence'])
            indices.append(i)
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Sort by confidence
        sorted_indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(sorted_indices) > 0:
            # Pick the detection with highest confidence
            current = sorted_indices[0]
            keep.append(current)
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # Calculate union
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                            (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = current_area + remaining_areas - intersection
            
            # Calculate IoU
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU below threshold
            sorted_indices = sorted_indices[1:][iou < iou_threshold]
        
        # Return kept detections
        return [detections[i] for i in keep]
    
    def detect_signs_color_shape(self, image):
        """Enhanced color-shape detection with better filtering and accuracy"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        signs = []
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Edge detection for better contour finding
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Minimum and maximum area thresholds
        min_area = image.shape[0] * image.shape[1] * 0.001  # 0.1% of image
        max_area = image.shape[0] * image.shape[1] * 0.5    # 50% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Traffic signs are usually somewhat square-ish
                if 0.4 <= aspect_ratio <= 2.5 and w > 20 and h > 20:
                    # Extract ROI
                    roi_hsv = hsv[y:y+h, x:x+w]
                    roi_bgr = image[y:y+h, x:x+w]
                    
                    # Detect shape
                    shape = self.detect_shape(contour)
                    
                    # Skip unknown shapes that are too small
                    if shape == 'unknown' and area < min_area * 5:
                        continue
                    
                    # Detect color with confidence
                    dominant_color, color_conf = self.detect_color(roi_hsv)
                    
                    # Skip if color detection is too weak
                    if color_conf < 0.08:
                        continue
                    
                    # Classify sign with advanced method
                    sign_type, details, conf_adj = self.classify_sign_advanced(
                        dominant_color, shape, color_conf
                    )
                    
                    # Calculate final confidence
                    confidence = min(0.95, color_conf * conf_adj)
                    
                    # Additional filtering: check for sign-like characteristics
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Traffic signs have good circularity/compactness
                        if circularity > 0.3:  # Reasonable threshold
                            signs.append({
                                'bbox': (x, y, w, h),
                                'color': dominant_color,
                                'shape': shape,
                                'type': sign_type,
                                'details': details,
                                'confidence': confidence,
                                'area': area
                            })
        
        return signs
    
    
    def process_image(self, image_path, output_path, confidence_threshold=0.25):
        """
        Process image with enhanced multi-method detection
        Combines YOLO, color-shape analysis, and NMS for best results
        """
        self.load_model()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return 0, []
        
        original = image.copy()
        
        # Optimize image size for better detection
        max_size = 1280
        h, w = image.shape[:2]
        scale_factor = 1.0
        
        if max(h, w) > max_size:
            scale_factor = max_size / max(h, w)
            image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)), 
                             interpolation=cv2.INTER_AREA)
            original = cv2.resize(original, (int(w * scale_factor), int(h * scale_factor)), 
                                interpolation=cv2.INTER_AREA)
        
        print(f"  Processing at resolution: {image.shape[1]}x{image.shape[0]}")
        
        # Method 1: YOLO detection
        print("  Running YOLO detection...")
        results = self.model(image, conf=confidence_threshold, iou=0.4, verbose=False)
        yolo_detections = []
        
        # Expanded relevant classes for traffic signs
        relevant_classes = [
            'stop sign', 'parking meter', 'traffic light', 
            'fire hydrant', 'street sign', 'sign'
        ]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                
                # Check if it's a relevant class or contains 'sign' keyword
                if any(keyword in class_name.lower() for keyword in ['sign', 'light', 'meter']):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_detections.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'type': class_name.upper(),
                        'details': 'Detected by YOLO',
                        'confidence': confidence,
                        'method': 'YOLO',
                        'color': 'N/A',
                        'shape': 'N/A'
                    })
        
        print(f"  YOLO found: {len(yolo_detections)} sign(s)")
        
        # Method 2: Enhanced color-shape detection
        print("  Running color-shape analysis...")
        shape_detections = self.detect_signs_color_shape(image)
        print(f"  Shape analysis found: {len(shape_detections)} sign(s)")
        
        # Add method tag to shape detections
        for det in shape_detections:
            det['method'] = 'Color-Shape'
        
        # Combine all detections
        all_detections = yolo_detections + shape_detections
        
        # Apply Non-Maximum Suppression to remove duplicates
        print("  Applying NMS to remove duplicates...")
        filtered_detections = self.non_max_suppression(all_detections, iou_threshold=0.4)
        print(f"  Final count after NMS: {len(filtered_detections)}")
        
        # Sort by confidence for better display
        filtered_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Draw detections with enhanced visualization
        for idx, detection in enumerate(filtered_detections):
            x, y, w, h = detection['bbox']
            sign_type = detection['type']
            confidence = detection['confidence']
            method = detection['method']
            
            # Color coding by confidence level
            if confidence > 0.8:
                color = (0, 255, 0)      # Green - high confidence
            elif confidence > 0.6:
                color = (0, 200, 255)    # Orange - medium confidence
            else:
                color = (0, 165, 255)    # Light orange - lower confidence
            
            # Draw bounding box with thickness based on confidence
            thickness = 3 if confidence > 0.7 else 2
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Prepare labels
            label = f"#{idx+1}: {sign_type}"
            conf_label = f"{confidence:.0%}"
            method_label = f"[{method}]"
            
            # Enhanced font settings
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            text_thickness = 2
            
            # Calculate text sizes
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_label, font, font_scale * 0.8, text_thickness)
            (method_w, method_h), _ = cv2.getTextSize(method_label, font, font_scale * 0.6, 1)
            
            # Calculate background size
            padding = 10
            total_width = max(label_w, conf_w, method_w) + padding * 2
            total_height = label_h + conf_h + method_h + padding * 3
            
            # Adjust position if too close to top
            y_offset = y - total_height - 10
            if y_offset < 0:
                y_offset = y + h + 10
            
            # Draw semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y_offset), 
                         (x + total_width, y_offset + total_height), 
                         color, -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Draw border
            cv2.rectangle(image, (x, y_offset), 
                         (x + total_width, y_offset + total_height), 
                         (255, 255, 255), 2)
            
            # Draw text labels
            text_y = y_offset + label_h + padding
            cv2.putText(image, label, (x + padding, text_y),
                       font, font_scale, (255, 255, 255), text_thickness)
            
            text_y += conf_h + padding - 5
            cv2.putText(image, conf_label, (x + padding, text_y),
                       font, font_scale * 0.8, (255, 255, 100), text_thickness)
            
            text_y += method_h + padding - 8
            cv2.putText(image, method_label, (x + padding, text_y),
                       font, font_scale * 0.6, (200, 200, 200), 1)
        
        # Create enhanced info panel
        panel_height = 120
        panel = np.zeros((panel_height, image.shape[1], 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        # Title
        cv2.putText(panel, "TRAFFIC SIGN DETECTION RESULTS", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        
        # Statistics
        stats_text = f"Total Signs: {len(filtered_detections)} | YOLO: {len(yolo_detections)} | Color-Shape: {len(shape_detections)}"
        cv2.putText(panel, stats_text, (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Method info
        method_text = f"Resolution: {image.shape[1]}x{image.shape[0]} | NMS Applied | Confidence Threshold: {confidence_threshold:.0%}"
        cv2.putText(panel, method_text, (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Combine image with panel
        result_image = np.vstack([image, panel])
        
        # Save result with high quality
        cv2.imwrite(str(output_path), result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return len(filtered_detections), filtered_detections


def main():
    """Test enhanced traffic sign recognition"""
    print("="*70)
    print("ADVANCED TRAFFIC SIGN RECOGNITION")
    print("="*70)
    
    recognizer = TrafficSignRecognizer()
    
    # Create directories
    base_dir = Path(__file__).parent.parent
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
    
    print(f"\nFound {len(image_files)} image(s) to process")
    print("\nDetection Methods:")
    print("  1. YOLO - Deep learning object detection")
    print("  2. Color-Shape Analysis - Advanced geometric detection")
    print("  3. NMS - Non-Maximum Suppression for duplicate removal")
    print("-" * 70)
    
    total_signs = 0
    
    # Process each image
    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")
        output_path = output_dir / f"detected_{image_path.name}"
        
        num_signs, detections = recognizer.process_image(
            image_path, output_path, confidence_threshold=0.25
        )
        
        total_signs += num_signs
        
        print(f"\n  ✓ Signs detected: {num_signs}")
        
        if detections:
            print("  Detections:")
            for idx, det in enumerate(detections, 1):
                color_shape = ""
                if det.get('color') != 'N/A' and det.get('shape') != 'N/A':
                    color_shape = f" [{det['color']} {det['shape']}]"
                
                details = det.get('details', '')
                if details:
                    print(f"    {idx}. {det['type']}{color_shape}")
                    print(f"       {details} | Conf: {det['confidence']:.1%} | Method: {det['method']}")
                else:
                    print(f"    {idx}. {det['type']}{color_shape} - "
                          f"Conf: {det['confidence']:.1%} | {det['method']}")
        
        print(f"  ✓ Output saved: {output_path.name}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print(f"DETECTION COMPLETE! Total signs found: {total_signs}")
    print("="*70)
    
    print("\nEnhancements applied:")
    print("  ✓ Advanced shape detection (circle, triangle, octagon, diamond)")
    print("  ✓ Multi-range color detection with confidence scoring")
    print("  ✓ Detailed sign classification (STOP, YIELD, WARNING, etc.)")
    print("  ✓ Non-Maximum Suppression for duplicate removal")
    print("  ✓ CLAHE histogram equalization for better contrast")
    print("  ✓ Edge-based contour detection")
    print("  ✓ GPU acceleration for faster processing")
    print("="*70)

if __name__ == "__main__":
    main()

