"""
Image utility functions for processing and manipulation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def resize_image(
    image: np.ndarray, 
    max_dimension: int = 1280,
    interpolation: int = cv2.INTER_AREA
) -> Tuple[np.ndarray, float]:
    """
    Resize image to max dimension while maintaining aspect ratio
    
    Args:
        image: Input image
        max_dimension: Maximum width or height
        interpolation: OpenCV interpolation method
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(image, new_size, interpolation=interpolation)
        return resized, scale
    return image, 1.0


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image or None if failed
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
    return image


def save_image(
    image: np.ndarray,
    output_path: Path,
    quality: int = 95
) -> bool:
    """
    Save image with optimized compression
    
    Args:
        image: Image to save
        output_path: Output file path
        quality: JPEG quality (0-100)
        
    Returns:
        True if successful
    """
    try:
        cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False
