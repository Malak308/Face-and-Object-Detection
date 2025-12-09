"""
Face Recognition and Emotion Detection
This script detects faces in images and analyzes emotions using DeepFace.
"""

import cv2
from deepface import DeepFace
import os
from pathlib import Path
import numpy as np

def detect_faces_and_emotions(image_path, output_path):
    """
    Detect faces and analyze emotions in an image (OPTIMIZED)
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"\n{'='*50}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*50}")
    
    try:
        # Optimize image size for faster processing
        max_size = 1280
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            temp_image = cv2.resize(image, (int(w * scale), int(h * scale)))
            temp_path = str(image_path).replace(image_path.suffix, '_temp' + image_path.suffix)
            cv2.imwrite(temp_path, temp_image)
            analysis_path = temp_path
            scale_factor = 1 / scale
        else:
            analysis_path = str(image_path)
            scale_factor = 1
        
        # Analyze faces with DeepFace (optimized settings)
        results = DeepFace.analyze(
            img_path=analysis_path,
            actions=['emotion', 'age', 'gender'],  # Removed 'race' for speed
            enforce_detection=False,
            detector_backend='opencv',  # Fastest detector
            silent=True
        )
        
        # Clean up temp file if created
        if scale_factor != 1:
            os.remove(analysis_path)
        
        # Handle both single face and multiple faces
        if not isinstance(results, list):
            results = [results]
        
        if len(results) == 0:
            print("No faces detected")
            cv2.imwrite(str(output_path), image)
            return
        
        print(f"Found {len(results)} face(s)")
        
        # Draw rectangles and labels for each face
        for idx, result in enumerate(results):
            # Get face region and scale back to original
            region = result['region']
            x, y, w, h = int(region['x'] * scale_factor), int(region['y'] * scale_factor), \
                         int(region['w'] * scale_factor), int(region['h'] * scale_factor)
            
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Get emotion with highest score
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            emotion_score = emotions[dominant_emotion]
            
            # Get other attributes
            age = result['age']
            gender = result['dominant_gender']
            gender_score = result['gender'][gender]
            
            # Print details
            print(f"\nFace {idx + 1}:")
            print(f"  Emotion: {dominant_emotion} ({emotion_score:.1f}%)")
            print(f"  Age: ~{age} years")
            print(f"  Gender: {gender} ({gender_score:.1f}%)")
            print(f"  Top 3 emotions:")
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    - {emotion}: {score:.1f}%")
            
            # Create label with emotion and confidence
            label = f"{dominant_emotion} {emotion_score:.0f}%"
            
            # Enhanced text rendering with larger fonts
            font = cv2.FONT_HERSHEY_DUPLEX  # Better font than SIMPLEX
            font_scale = 1.0  # Increased from 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw enhanced background rectangle for text with padding
            padding = 8
            cv2.rectangle(image, 
                         (x - padding, y - text_height - 20), 
                         (x + text_width + padding * 2, y - 2),
                         (0, 200, 0), -1)  # Filled background
            cv2.rectangle(image, 
                         (x - padding, y - text_height - 20), 
                         (x + text_width + padding * 2, y - 2),
                         (0, 255, 0), 2)  # Border
            
            # Draw emotion label with better contrast
            cv2.putText(image, label, (x + 2, y - 10),
                       font, font_scale, (255, 255, 255), thickness + 1)  # White text with outline
            
            # Draw age and gender below face with background
            info_label = f"{gender}, {int(age)}y"
            info_font_scale = 0.8  # Increased from 0.6
            (info_w, info_h), _ = cv2.getTextSize(info_label, font, info_font_scale, 2)
            cv2.rectangle(image, (x - padding, y + h + 8), (x + info_w + padding * 2, y + h + info_h + 18), (0, 200, 0), -1)
            cv2.rectangle(image, (x - padding, y + h + 8), (x + info_w + padding * 2, y + h + info_h + 18), (0, 255, 0), 2)
            cv2.putText(image, info_label, (x + 2, y + h + 25),
                       font, info_font_scale, (255, 255, 255), 2)
        
        # Save result with optimized compression
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\nOutput saved to: {output_path}")
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        cv2.imwrite(str(output_path), image)

def main():
    """Main function to run face and emotion detection"""
    
    print("="*60)
    print("FACE RECOGNITION & EMOTION DETECTION")
    print("="*90)
    
    # Define paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / "input_images"
    output_dir = base_dir / "output_images" / "emotions"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Get all images
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"\n⚠ No images found in {input_dir}")
        print("Please add images with faces to the 'input_images' folder.")
        return
    
    print(f"\nFound {len(image_files)} image(s) to process")
    print("\nNote: This will download models on first run (~100MB)")
    print("Processing may take a moment...\n")
    
    # Process each image
    for image_path in image_files:
        output_path = output_dir / f"emotion_{image_path.name}"
        detect_faces_and_emotions(image_path, output_path)
    
    print("\n" + "="*60)
    print("EMOTION DETECTION COMPLETE!")
    print(f"All results saved in: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
