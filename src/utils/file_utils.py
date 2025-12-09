"""
File system utility functions
"""

from pathlib import Path
from typing import List, Set


# Supported image formats
SUPPORTED_FORMATS: Set[str] = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
}


def ensure_directory(directory: Path) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path to create
    """
    directory.mkdir(parents=True, exist_ok=True)


def get_supported_images(directory: Path) -> List[Path]:
    """
    Get all supported image files in a directory
    
    Args:
        directory: Directory to search
        
    Returns:
        List of image file paths
    """
    if not directory.exists():
        return []
    
    return [
        f for f in directory.iterdir()
        if f.suffix.lower() in SUPPORTED_FORMATS
    ]


def setup_project_directories(base_dir: Path) -> dict:
    """
    Setup all required project directories
    
    Args:
        base_dir: Base project directory
        
    Returns:
        Dictionary of created directory paths
    """
    directories = {
        'input': {
            'objects': base_dir / 'input_images' / 'objects',
            'faces': base_dir / 'input_images' / 'faces',
            'classification': base_dir / 'input_images' / 'classification',
            'segmentation': base_dir / 'input_images' / 'segmentation',
            'traffic_signs': base_dir / 'input_images' / 'traffic_signs'
        },
        'output': {
            'objects': base_dir / 'output_images' / 'objects',
            'emotions': base_dir / 'output_images' / 'emotions',
            'classification': base_dir / 'output_images' / 'classification',
            'segmentation': base_dir / 'output_images' / 'segmentation',
            'traffic_signs': base_dir / 'output_images' / 'traffic_signs'
        }
    }
    
    # Create all directories
    for category in directories.values():
        for directory in category.values():
            ensure_directory(directory)
    
    return directories
