"""
Configuration file for Computer Vision Application
Contains optimized settings for all detection modes
"""

# Performance Settings
PERFORMANCE = {
    'max_image_dimension': 1280,  # Maximum dimension for processing
    'jpeg_quality': 95,  # Output image quality
    'enable_caching': True,  # Enable model caching
    'use_gpu': True,  # Use GPU if available
    'num_threads': 4,  # Number of threads for processing
}

# Object Detection Settings (YOLO)
OBJECT_DETECTION = {
    'model': 'yolov8n.pt',  # Can be: yolov8n, yolov8s, yolov8m for better accuracy
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
    'max_detections': 100,
    'optimize_for_speed': True,
}

# Face & Emotion Detection Settings
FACE_EMOTION = {
    'detector_backend': 'opencv',  # opencv, retinaface, mtcnn, ssd
    'actions': ['emotion', 'age', 'gender'],  # Remove 'race' for speed
    'enforce_detection': False,
    'silent': True,
    'max_image_size': 1280,
}

# Image Classification Settings
IMAGE_CLASSIFICATION = {
    'model': 'ResNet50',  # ResNet50, VGG16, InceptionV3
    'top_predictions': 5,
    'batch_size': 1,
    'input_size': (224, 224),
}

# Image Segmentation Settings
SEGMENTATION = {
    'algorithm': 'simple',  # simple, watershed, kmeans
    'min_segment_area': 0.001,  # Percentage of image
    'edge_threshold_low': 30,
    'edge_threshold_high': 100,
    'alpha_blend': 0.5,
}

# Traffic Sign Recognition Settings
TRAFFIC_SIGN = {
    'yolo_confidence': 0.3,
    'color_detection_threshold': 0.1,
    'min_sign_area': 500,
    'use_yolo': True,
    'use_color_shape': True,
}

# GUI Settings
GUI = {
    'appearance_mode': 'dark',  # dark, light, system
    'color_theme': 'blue',  # blue, green, dark-blue
    'window_size': '1500x900',
    'min_window_size': (1400, 850),
    'update_interval': 100,  # ms
}

# File Settings
FILES = {
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
    'max_file_size_mb': 50,
    'auto_save': True,
}

# Advanced Settings
ADVANCED = {
    'verbose_logging': False,
    'show_processing_time': True,
    'enable_batch_processing': False,
    'save_metadata': False,
}
