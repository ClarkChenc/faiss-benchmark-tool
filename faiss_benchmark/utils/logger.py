"""
Logging utilities for FAISS benchmark framework
"""

import logging
import sys
from typing import Optional
from datetime import datetime


class Logger:
    """Logger utility for FAISS benchmark framework"""
    
    def __init__(self, name: str = "faiss_benchmark", level: str = "INFO", 
                 log_file: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)
    
    def log_benchmark_start(self, dataset_name: str, index_name: str) -> None:
        """Log benchmark start"""
        self.info(f"Starting benchmark: {dataset_name} with {index_name}")
    
    def log_benchmark_end(self, dataset_name: str, index_name: str, 
                         duration: float) -> None:
        """Log benchmark completion"""
        self.info(f"Completed benchmark: {dataset_name} with {index_name} "
                 f"in {duration:.2f} seconds")
    
    def log_index_build(self, index_name: str, build_time: float, 
                       n_vectors: int) -> None:
        """Log index building"""
        self.info(f"Built {index_name} index with {n_vectors} vectors "
                 f"in {build_time:.2f} seconds")
    
    def log_search_performance(self, index_name: str, qps: float, 
                              recall: float, k: int) -> None:
        """Log search performance"""
        self.info(f"{index_name} - QPS: {qps:.2f}, Recall@{k}: {recall:.4f}")
    
    def log_memory_usage(self, stage: str, memory_mb: float) -> None:
        """Log memory usage"""
        self.info(f"Memory usage at {stage}: {memory_mb:.2f} MB")
    
    def log_gpu_info(self, gpu_name: str, memory_total: float, 
                     memory_used: float) -> None:
        """Log GPU information"""
        self.info(f"GPU: {gpu_name}, Memory: {memory_used:.2f}/{memory_total:.2f} MB")


# Global logger instance
_global_logger = None


def get_logger(name: str = "faiss_benchmark", level: str = "INFO", 
               log_file: Optional[str] = None) -> Logger:
    """
    Get global logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(name, level, log_file)
    return _global_logger