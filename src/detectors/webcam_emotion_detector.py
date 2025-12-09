"""
Real-time Emotion Detection from Webcam
Uses DeepFace for emotion analysis with live video feed
"""

import cv2
import numpy as np
from deepface import DeepFace
from pathlib import Path
import time

class WebcamEmotionDetector:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'surprise': (255, 255, 0), # Cyan
            'fear': (255, 0, 255),     # Magenta
            'disgust': (0, 128, 255),  # Orange
            'neutral': (200, 200, 200) # Gray
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def analyze_frame(self, frame):
        """Analyze a single frame for emotions"""
        try:
            # Analyze with DeepFace
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            
            # Handle single or multiple faces
            if isinstance(result, list):
                return result
            else:
                return [result]
        except Exception as e:
            return []
    
    def draw_emotion_info(self, frame, face_data):
        """Draw emotion information on frame"""
        for face in face_data:
            # Get face region
            region = face.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            
            # Get dominant emotion
            emotions = face.get('emotion', {})
            if emotions:
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion]
                
                # Get color for emotion
                color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Prepare label
                label = f"{dominant_emotion.upper()}"
                conf_text = f"{confidence:.1f}%"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                cv2.rectangle(frame, (x, y - 50), (x + max(label_size[0], w), y), color, -1)
                
                # Draw text
                cv2.putText(frame, label, (x + 5, y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, conf_text, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw emotion bars on the side
                bar_x = x + w + 10
                bar_y_start = y
                bar_height = 15
                
                for idx, (emotion, value) in enumerate(sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]):
                    bar_y = bar_y_start + idx * (bar_height + 5)
                    bar_width = int(value * 2)  # Scale to max 200 pixels
                    
                    # Draw bar
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                                 self.emotion_colors.get(emotion, (255, 255, 255)), -1)
                    
                    # Draw emotion name
                    cv2.putText(frame, f"{emotion}: {value:.0f}%", 
                               (bar_x, bar_y + bar_height - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def start_webcam(self, camera_index=0):
        """Start webcam emotion detection"""
        print("="*60)
        print("WEBCAM EMOTION DETECTION")
        print("="*60)
        print("\nStarting webcam...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'p' to pause/resume")
        print("-"*60)
        
        # Open webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Webcam started successfully!")
        print("\nAnalyzing emotions in real-time...\n")
        
        # Create output directory
        output_dir = Path(__file__).parent.parent / "output_images" / "emotions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        analyze_interval = 10  # Analyze every 10 frames for better performance
        last_analysis = None
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Failed to capture frame")
                    break
                
                # Create display frame
                display_frame = frame.copy()
                
                # Analyze periodically
                if frame_count % analyze_interval == 0:
                    last_analysis = self.analyze_frame(frame)
                
                # Draw emotion info if available
                if last_analysis:
                    display_frame = self.draw_emotion_info(display_frame, last_analysis)
                
                # Add info overlay
                info_text = "LIVE EMOTION DETECTION | Press 'q' to quit | 's' to save | 'p' to pause"
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
                # Enhanced info text with background
                (info_w, info_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
                cv2.rectangle(display_frame, (5, 5), (info_w + 20, info_h + 15), (40, 40, 40), -1)
                cv2.rectangle(display_frame, (5, 5), (info_w + 20, info_h + 15), (0, 255, 255), 2)
                cv2.putText(display_frame, info_text, (10, 25),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)  # Increased from 0.6
                
                frame_count += 1
            else:
                # Paused - show pause indicator
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 100), -1)
                pause_text = "PAUSED | Press 'p' to resume | 'q' to quit"
                (pause_w, pause_h), _ = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
                cv2.rectangle(display_frame, (5, 5), (pause_w + 20, pause_h + 15), (0, 0, 139), -1)
                cv2.rectangle(display_frame, (5, 5), (pause_w + 20, pause_h + 15), (255, 255, 255), 2)
                cv2.putText(display_frame, pause_text, (10, 25),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)  # Increased from 0.6
            
            # Show frame
            cv2.imshow('Webcam Emotion Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nStopping webcam...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = output_dir / f"webcam_{timestamp}.jpg"
                cv2.imwrite(str(save_path), display_frame)
                print(f"✓ Frame saved: {save_path.name}")
            elif key == ord('p'):
                # Toggle pause
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"▶ {status}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("WEBCAM STOPPED")
        print("="*60)

def main():
    """Run webcam emotion detection"""
    detector = WebcamEmotionDetector()
    detector.start_webcam()

if __name__ == "__main__":
    main()
