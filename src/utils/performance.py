"""
Performance monitoring utility for Computer Vision Application
Tracks processing time, memory usage, and model performance
"""

import time
import psutil
from functools import wraps
from datetime import datetime


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_images_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'detection_counts': {},
            'errors': 0
        }
        self.start_time = None
        self.process = psutil.Process()
    
    def start_timer(self):
        """Start timing an operation"""
        self.start_time = time.time()
    
    def stop_timer(self):
        """Stop timing and return elapsed time"""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed
    
    def record_processing(self, processing_time, mode, num_detections=0):
        """Record processing metrics"""
        self.metrics['total_images_processed'] += 1
        self.metrics['total_processing_time'] += processing_time
        self.metrics['average_processing_time'] = (
            self.metrics['total_processing_time'] / 
            self.metrics['total_images_processed']
        )
        
        if mode not in self.metrics['detection_counts']:
            self.metrics['detection_counts'][mode] = {'count': 0, 'total_detections': 0}
        
        self.metrics['detection_counts'][mode]['count'] += 1
        self.metrics['detection_counts'][mode]['total_detections'] += num_detections
    
    def record_error(self):
        """Record an error"""
        self.metrics['errors'] += 1
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self):
        """Get current CPU usage percentage"""
        return self.process.cpu_percent(interval=0.1)
    
    def get_summary(self):
        """Get performance summary"""
        return {
            'images_processed': self.metrics['total_images_processed'],
            'avg_time': f"{self.metrics['average_processing_time']:.2f}s",
            'total_time': f"{self.metrics['total_processing_time']:.2f}s",
            'memory_mb': f"{self.get_memory_usage():.1f}MB",
            'cpu_percent': f"{self.get_cpu_usage():.1f}%",
            'errors': self.metrics['errors'],
            'by_mode': self.metrics['detection_counts']
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'total_images_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'detection_counts': {},
            'errors': 0
        }


def time_it(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️ {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def log_performance(message, start_time=None):
    """Log a performance message with optional timing"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if start_time:
        elapsed = time.time() - start_time
        print(f"[{timestamp}] {message} ({elapsed:.2f}s)")
    else:
        print(f"[{timestamp}] {message}")
