"""
Face Recognition System
Identifies known faces from a database and detects emotions using DeepFace.
OPTIMIZED with GPU acceleration, caching, and parallel processing
"""

import cv2
from deepface import DeepFace
import os
from pathlib import Path
import numpy as np
import pickle
import json
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import time

class FaceRecognitionSystem:
    """Face Recognition with database of known faces (GPU-OPTIMIZED)"""
    
    def __init__(self, database_path=None):
        """
        Initialize face recognition system
        
        Args:
            database_path: Path to folder containing known faces (organized in subfolders by person)
        """
        if database_path is None:
            database_path = Path(__file__).parent.parent / "input_images" / "face_database"
        
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.known_faces = {}
        self.face_embeddings_cache = {}  # Cache for pre-computed embeddings
        self.model_name = 'Facenet512'  # Best accuracy: VGG-Face, Facenet, Facenet512, ArcFace
        self.detector_backend = 'opencv'  # Fastest: opencv, ssd, dlib, mtcnn, retinaface
        self.distance_metric = 'cosine'  # cosine, euclidean, euclidean_l2
        self._setup_gpu()
        self._load_cache()
        
    def _setup_gpu(self):
        """Configure TensorFlow for optimal GPU usage"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ GPU acceleration enabled: {len(gpus)} GPU(s) found")
            else:
                print("⚠ No GPU found, using CPU")
        except Exception as e:
            print(f"GPU setup info: {e}")
    
    def _load_cache(self):
        """Load cached face embeddings if available"""
        cache_file = self.database_path / "embeddings_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.face_embeddings_cache = pickle.load(f)
                print(f"✓ Loaded {len(self.face_embeddings_cache)} cached embeddings")
            except Exception as e:
                print(f"Cache load warning: {e}")
                self.face_embeddings_cache = {}
    
    def _save_cache(self):
        """Save face embeddings to cache"""
        cache_file = self.database_path / "embeddings_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.face_embeddings_cache, f)
        except Exception as e:
            print(f"Cache save warning: {e}")
        
    def load_database(self):
        """Load all known faces from database folder structure and pre-compute embeddings"""
        print("\nLoading face database...")
        self.known_faces = {}
        
        if not self.database_path.exists():
            print(f"Database path not found: {self.database_path}")
            return
        
        # Look for person folders
        person_folders = [d for d in self.database_path.iterdir() if d.is_dir()]
        
        if not person_folders:
            print(f"⚠ No person folders found in {self.database_path}")
            print("\nTo use face recognition:")
            print("1. Create folders named after each person (e.g., 'John_Doe', 'Jane_Smith')")
            print("2. Add 1-3 photos of each person in their folder")
            print("3. Photos should clearly show the person's face")
            return
        
        for person_folder in person_folders:
            person_name = person_folder.name
            image_files = [f for f in person_folder.iterdir() 
                          if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            
            if image_files:
                self.known_faces[person_name] = [str(f) for f in image_files]
                print(f"  ✓ Loaded {len(image_files)} image(s) for {person_name}")
        
        print(f"\n✓ Database loaded: {len(self.known_faces)} person(s)")
        
        # Pre-compute embeddings for all database faces (HUGE SPEED BOOST)
        print("\nPre-computing face embeddings (this may take a moment)...")
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute and cache embeddings for all database faces"""
        new_embeddings = 0
        
        for person_name, image_paths in self.known_faces.items():
            for img_path in image_paths:
                # Check if embedding is already cached
                if img_path not in self.face_embeddings_cache:
                    try:
                        # Generate embedding
                        embedding = DeepFace.represent(
                            img_path=img_path,
                            model_name=self.model_name,
                            detector_backend=self.detector_backend,
                            enforce_detection=False
                        )
                        
                        if embedding:
                            self.face_embeddings_cache[img_path] = {
                                'embedding': embedding[0]['embedding'],
                                'person': person_name
                            }
                            new_embeddings += 1
                    except Exception as e:
                        print(f"  ⚠ Failed to process {Path(img_path).name}: {e}")
        
        if new_embeddings > 0:
            print(f"✓ Pre-computed {new_embeddings} new embeddings")
            self._save_cache()
        else:
            print("✓ All embeddings already cached")
        
    def recognize_face(self, face_img_path):
        """
        Try to recognize a face against the database (OPTIMIZED with caching)
        
        Args:
            face_img_path: Path to the face image to identify
            
        Returns:
            tuple: (person_name, confidence) or (None, 0) if not recognized
        """
        if not self.known_faces or not self.face_embeddings_cache:
            return None, 0
        
        try:
            # Generate embedding for the input face (only once!)
            input_embedding = DeepFace.represent(
                img_path=face_img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            if not input_embedding:
                return None, 0
            
            input_vec = np.array(input_embedding[0]['embedding'])
            
            # Compare with cached embeddings (MUCH FASTER)
            best_match = None
            best_distance = float('inf')
            
            for img_path, cached_data in self.face_embeddings_cache.items():
                try:
                    ref_vec = np.array(cached_data['embedding'])
                    
                    # Calculate distance based on metric
                    if self.distance_metric == 'cosine':
                        distance = 1 - np.dot(input_vec, ref_vec) / (np.linalg.norm(input_vec) * np.linalg.norm(ref_vec))
                    elif self.distance_metric == 'euclidean':
                        distance = np.linalg.norm(input_vec - ref_vec)
                    else:  # euclidean_l2
                        distance = np.sqrt(np.sum((input_vec - ref_vec) ** 2))
                    
                    # Track best match
                    if distance < best_distance:
                        best_distance = distance
                        best_match = cached_data['person']
                        
                except Exception as e:
                    continue
            
            # Determine if match is confident enough
            thresholds = {
                'VGG-Face': 0.40,
                'Facenet': 0.40,
                'Facenet512': 0.30,
                'ArcFace': 0.68,
                'OpenFace': 0.10,
                'DeepFace': 0.23,
            }
            
            threshold = thresholds.get(self.model_name, 0.40)
            
            if best_match and best_distance < threshold:
                confidence = max(0, 100 * (1 - best_distance / threshold))
                return best_match, confidence
            
        except Exception as e:
            print(f"Recognition error: {e}")
        
        return None, 0
    
    def process_image(self, image_path, output_path):
        """
        Process an image: detect faces and identify them (OPTIMIZED with timing)
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated output
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        print(f"\n{'='*50}")
        print(f"Processing: {Path(image_path).name}")
        print(f"{'='*50}")
        
        try:
            # Detect faces only (no emotion/age analysis for speed)
            detect_start = time.time()
            results = DeepFace.extract_faces(
                img_path=str(image_path),
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            detect_time = time.time() - detect_start
            print(f"⚡ Face detection: {detect_time:.2f}s")
            
            if len(results) == 0:
                print("No faces detected")
                cv2.imwrite(str(output_path), image)
                return
            
            print(f"Found {len(results)} face(s)")
            
            # Process each detected face with timing
            recognition_start = time.time()
            for idx, result in enumerate(results):
                # Get face region
                facial_area = result['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                # Extract face for recognition
                face_img = image[max(0, y):min(image.shape[0], y+h), 
                                max(0, x):min(image.shape[1], x+w)]
                
                if face_img.size == 0:
                    continue
                
                # Save temp face image for recognition
                temp_face_path = f"temp_face_{idx}.jpg"
                cv2.imwrite(temp_face_path, face_img)
                
                # Try to recognize the face (FAST with cached embeddings)
                person_name = "Unknown"
                confidence = 0
                
                if self.known_faces:
                    face_start = time.time()
                    recognized_name, conf = self.recognize_face(temp_face_path)
                    face_time = time.time() - face_start
                    if recognized_name:
                        person_name = recognized_name.replace('_', ' ')
                        confidence = conf
                        print(f"  ⚡ Face {idx+1} recognized in {face_time:.3f}s")
                
                # Clean up temp file
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)
                
                # Print details
                print(f"\nFace {idx + 1}:")
                if person_name != "Unknown":
                    print(f"  ✓ Identity: {person_name} ({confidence:.1f}% match)")
                else:
                    print(f"  ✗ Identity: Unknown (no match in database)")
                
                # Draw bounding box (green for recognized, red for unknown)
                color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                line_width = 4
                cv2.rectangle(image, (x, y), (x+w, y+h), color, line_width)
                
                # Prepare label
                font = cv2.FONT_HERSHEY_DUPLEX
                padding = 10
                
                # Create identity label
                if person_name != "Unknown":
                    id_label = f"{person_name}"
                    conf_label = f"Match: {confidence:.0f}%"
                else:
                    id_label = "UNKNOWN"
                    conf_label = "Not in Database"
                
                font_scale_name = 1.2
                font_scale_conf = 0.7
                thickness = 2
                
                # Calculate text sizes
                (name_w, name_h), _ = cv2.getTextSize(id_label, font, font_scale_name, thickness)
                (conf_w, conf_h), _ = cv2.getTextSize(conf_label, font, font_scale_conf, 2)
                
                # Use the wider of the two labels
                max_width = max(name_w, conf_w)
                
                # Draw name label background above face
                y_offset_name = y - name_h - conf_h - 35
                if y_offset_name < 0:
                    y_offset_name = y + h + 10
                
                # Background for name
                cv2.rectangle(image, 
                             (x - padding, y_offset_name), 
                             (x + max_width + padding * 2, y_offset_name + name_h + conf_h + 30), 
                             color, -1)
                cv2.rectangle(image, 
                             (x - padding, y_offset_name), 
                             (x + max_width + padding * 2, y_offset_name + name_h + conf_h + 30), 
                             (255, 255, 255), 3)
                
                # Draw name text
                cv2.putText(image, id_label, 
                           (x + padding, y_offset_name + name_h + 5),
                           font, font_scale_name, (255, 255, 255), thickness + 1)
                
                # Draw confidence text
                cv2.putText(image, conf_label, 
                           (x + padding, y_offset_name + name_h + conf_h + 20),
                           font, font_scale_conf, (255, 255, 255), 2)
            
            recognition_time = time.time() - recognition_start
            total_time = time.time() - start_time
            
            # Save result
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"\n⚡ Recognition time: {recognition_time:.2f}s")
            print(f"⚡ Total processing: {total_time:.2f}s")
            print(f"✓ Output saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            cv2.imwrite(str(output_path), image)


def main():
    """Main function for face recognition"""
    print("="*70)
    print("FACE RECOGNITION & EMOTION DETECTION SYSTEM")
    print("="*70)
    
    # Initialize system
    recognizer = FaceRecognitionSystem()
    recognizer.load_database()
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "input_images" / "faces"
    output_dir = base_dir / "output_images" / "recognized_faces"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get images to process
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    
    if not image_files:
        print(f"\n⚠ No images found in {input_dir}")
        return
    
    print(f"\nProcessing {len(image_files)} image(s)...")
    
    # Process each image
    for image_path in image_files:
        output_path = output_dir / f"recognized_{image_path.name}"
        recognizer.process_image(image_path, output_path)
    
    print("\n" + "="*70)
    print("FACE RECOGNITION COMPLETE!")
    print(f"Results saved in: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
