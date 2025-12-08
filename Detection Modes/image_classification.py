"""
Image Classification using ResNet50
Classifies images into 1000 ImageNet categories
"""

import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

class ImageClassifier:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        """Load pre-trained ResNet50 model"""
        if self.model is None:
            print("Loading ResNet50 model...")
            self.model = ResNet50(weights='imagenet')
            print("✓ Model loaded")
    
    def classify_image(self, image_path, top_k=5):
        """
        Classify an image
        
        Args:
            image_path: Path to image
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples
        """
        self.load_model()
        
        # Load and preprocess image
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        decoded = decode_predictions(predictions, top=top_k)[0]
        
        # Format results
        results = [(class_name, float(prob)) for (_, class_name, prob) in decoded]
        return results
    
    def process_image_with_labels(self, image_path, output_path):
        """Process image and save with classification labels"""
        # Get classifications
        results = self.classify_image(image_path, top_k=5)
        
        # Load original image
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        
        # Create overlay
        overlay_height = 250
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        overlay[:] = (40, 40, 40)  # Dark gray background
        
        # Add title
        cv2.putText(overlay, "IMAGE CLASSIFICATION", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.line(overlay, (10, 50), (w-10, 50), (0, 255, 255), 2)
        
        # Add top predictions
        y_offset = 85
        for idx, (class_name, prob) in enumerate(results, 1):
            # Class name
            text = f"{idx}. {class_name.replace('_', ' ').title()}"
            cv2.putText(overlay, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int((w - 250) * prob)
            cv2.rectangle(overlay, (230, y_offset-20), (230 + bar_width, y_offset-5),
                         (0, 255, 100), -1)
            
            # Percentage
            cv2.putText(overlay, f"{prob:.1%}", (w-120, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            y_offset += 40
        
        # Combine original image with overlay
        result = np.vstack([image, overlay])
        
        # Save result
        cv2.imwrite(str(output_path), result)
        
        return results

def main():
    """Test image classification"""
    print("="*60)
    print("IMAGE CLASSIFICATION")
    print("="*60)
    
    classifier = ImageClassifier()
    
    # Create directories
    base_dir = Path(__file__).parent
    input_dir = base_dir / "input_images" / "classification"
    output_dir = base_dir / "output_images" / "classification"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"\n⚠ No images found in {input_dir}")
        print("Please add images to classify.")
        return
    
    print(f"\nFound {len(image_files)} image(s) to classify\n")
    
    # Process each image
    for image_path in image_files:
        print(f"Processing: {image_path.name}")
        output_path = output_dir / f"classified_{image_path.name}"
        
        results = classifier.process_image_with_labels(image_path, output_path)
        
        print(f"\nTop 5 Predictions:")
        for idx, (class_name, prob) in enumerate(results, 1):
            print(f"  {idx}. {class_name.replace('_', ' ').title():30s} {prob:.1%}")
        print(f"\nOutput saved: {output_path.name}")
        print("-" * 60)
    
    print("\n" + "="*60)
    print("CLASSIFICATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
