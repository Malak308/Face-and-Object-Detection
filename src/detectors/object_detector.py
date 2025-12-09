"""
Object Detection using YOLOv8
This script performs real-time object detection on images using the YOLOv8 model.
It processes images from the input_images folder and saves annotated results to output_images.
"""

import cv2
from ultralytics import YOLO
import os
from pathlib import Path

def detect_objects(image_path, output_path, model, confidence_threshold=0.5):
    """
    Optimized object detection with performance enhancements
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Optimize image size for faster processing while maintaining quality
    max_dimension = 1280
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # Run detection with optimized parameters
    results = model(image, conf=confidence_threshold, iou=0.45, max_det=100, verbose=False)
    
    # Get the annotated image
    annotated_image = results[0].plot()
    
    # Save the result with optimized compression
    cv2.imwrite(str(output_path), annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Print detection summary
    detections = results[0].boxes
    print(f"\n{'='*50}")
    print(f"Image: {image_path.name}")
    print(f"{'='*50}")
    print(f"Total objects detected: {len(detections)}")
    
    if len(detections) > 0:
        print("\nDetected objects:")
        # Group by class for cleaner output
        class_counts = {}
        for box in detections:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(confidence)
        
        for class_name, confidences in sorted(class_counts.items()):
            avg_conf = sum(confidences) / len(confidences)
            print(f"  - {class_name}: {len(confidences)} detected (avg {avg_conf:.2%} confidence)")
    
    print(f"Output saved to: {output_path}")
    print(f"{'='*50}\n")

def main():
    """Main function to run object detection on all images in input folder"""
    
    print("="*60)
    print("OBJECT DETECTION WITH YOLOv8")
    print("="*60)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "input_images"
    output_dir = base_dir / "output_images"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Load YOLOv8 model (will download on first run)
    print("\nLoading YOLOv8 model...")
    try:
        # Use YOLOv8n for speed, or YOLOv8s for better accuracy
        model = YOLO('yolov8n.pt')
        # Enable half-precision for faster inference on supported hardware
        if hasattr(model, 'half'):
            model.half()
        print("Model loaded successfully! (Optimized for performance)")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative loading method...")
        model = YOLO('yolov8n.pt')
        print("Model loaded successfully!")
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Get all images in input directory
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"\n⚠ No images found in {input_dir}")
        print("Please add some images to the 'input_images' folder and run again.")
        return
    
    print(f"\nFound {len(image_files)} image(s) to process")
    
    # Process each image with lower threshold for more detections
    for image_path in image_files:
        output_path = output_dir / f"detected_{image_path.name}"
        detect_objects(image_path, output_path, model, confidence_threshold=0.15)
    
    print("\n" + "="*60)
    print("DETECTION COMPLETE!")
    print(f"All results saved in: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
