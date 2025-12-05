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
    Detect faces and analyze emotions in an image
    
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
        # Analyze faces with DeepFace (optimized for speed)
        results = DeepFace.analyze(
            img_path=str(image_path),
            actions=['emotion'],  # Only emotion analysis - much faster
            enforce_detection=False,
            detector_backend='opencv',  # Faster than default
            silent=True
        )
        
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
            # Get face region
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get emotion with highest score
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            emotion_score = emotions[dominant_emotion]
            
            # Print details
            print(f"\nFace {idx + 1}:")
            print(f"  Emotion: {dominant_emotion} ({emotion_score:.1f}%)")
            print(f"  All emotions:")
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {emotion}: {score:.1f}%")
            
            # Create label with emotion and confidence
            label = f"{dominant_emotion} {emotion_score:.0f}%"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(image, 
                         (x, y - text_height - 10), 
                         (x + text_width + 10, y),
                         (0, 255, 0), -1)
            
            # Draw emotion label
            cv2.putText(image, label, (x + 5, y - 5),
                       font, font_scale, (0, 0, 0), thickness)
        
        # Save result
        cv2.imwrite(str(output_path), image)
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
