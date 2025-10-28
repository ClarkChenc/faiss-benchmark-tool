"""
Metrics calculation for FAISS benchmark framework
"""

import time
import psutil
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..utils.logger import get_logger


class MetricsCalculator:
    """Calculator for various benchmark metrics"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        self.logger = get_logger()
    
    def calculate_recall(self, predicted_indices: np.ndarray, 
                        ground_truth: np.ndarray, k: int) -> float:
        """
        Calculate recall@k metric.
        
        Args:
            predicted_indices: Predicted nearest neighbor indices (n_queries, k_pred)
            ground_truth: Ground truth indices (n_queries, k_gt)
            k: Number of top results to consider
            
        Returns:
            Recall@k value
        """
        if predicted_indices.shape[0] != ground_truth.shape[0]:
            raise ValueError("Number of queries must match between predicted and ground truth")
        
        n_queries = predicted_indices.shape[0]
        k_pred = min(k, predicted_indices.shape[1])
        k_gt = min(k, ground_truth.shape[1])
        
        total_relevant = 0
        total_retrieved_relevant = 0
        
        for i in range(n_queries):
            # Get top-k predictions and ground truth
            pred_k = set(predicted_indices[i, :k_pred])
            gt_k = set(ground_truth[i, :k_gt])
            
            # Calculate intersection
            intersection = pred_k.intersection(gt_k)
            
            total_retrieved_relevant += len(intersection)
            total_relevant += len(gt_k)
        
        if total_relevant == 0:
            return 0.0
        
        recall = total_retrieved_relevant / total_relevant
        return recall
    
    def calculate_precision(self, predicted_indices: np.ndarray,
                           ground_truth: np.ndarray, k: int) -> float:
        """
        Calculate precision@k metric.
        
        Args:
            predicted_indices: Predicted nearest neighbor indices (n_queries, k_pred)
            ground_truth: Ground truth indices (n_queries, k_gt)
            k: Number of top results to consider
            
        Returns:
            Precision@k value
        """
        if predicted_indices.shape[0] != ground_truth.shape[0]:
            raise ValueError("Number of queries must match between predicted and ground truth")
        
        n_queries = predicted_indices.shape[0]
        k_pred = min(k, predicted_indices.shape[1])
        k_gt = ground_truth.shape[1]
        
        total_retrieved = 0
        total_retrieved_relevant = 0
        
        for i in range(n_queries):
            # Get top-k predictions and all ground truth
            pred_k = set(predicted_indices[i, :k_pred])
            gt_all = set(ground_truth[i, :k_gt])
            
            # Calculate intersection
            intersection = pred_k.intersection(gt_all)
            
            total_retrieved_relevant += len(intersection)
            total_retrieved += len(pred_k)
        
        if total_retrieved == 0:
            return 0.0
        
        precision = total_retrieved_relevant / total_retrieved
        return precision
    
    def calculate_qps(self, n_queries: int, search_time: float) -> float:
        """
        Calculate queries per second (QPS).
        
        Args:
            n_queries: Number of queries
            search_time: Total search time in seconds
            
        Returns:
            QPS value
        """
        if search_time <= 0:
            return 0.0
        
        return n_queries / search_time
    
    def calculate_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        """
        Calculate latency statistics.
        
        Args:
            latencies: List of individual query latencies
            
        Returns:
            Dictionary with latency statistics
        """
        if not latencies:
            return {
                'mean': 0.0,
                'median': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        latencies_array = np.array(latencies)
        
        return {
            'mean': float(np.mean(latencies_array)),
            'median': float(np.median(latencies_array)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'std': float(np.std(latencies_array))
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """
        Get current CPU usage.
        
        Returns:
            Dictionary with CPU usage information
        """
        return {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True)
        }
    
    def get_gpu_memory_usage(self) -> Optional[Dict[str, float]]:
        """
        Get GPU memory usage (requires pynvml).
        
        Returns:
            Dictionary with GPU memory usage or None if not available
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                'total_mb': memory_info.total / (1024 * 1024),
                'used_mb': memory_info.used / (1024 * 1024),
                'free_mb': memory_info.free / (1024 * 1024),
                'percent': (memory_info.used / memory_info.total) * 100
            }
            
        except ImportError:
            self.logger.warning("pynvml not available - cannot get GPU memory usage")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory usage: {e}")
            return None
    
    def benchmark_search_performance(self, index, queries: np.ndarray, 
                                   ground_truth: np.ndarray, k: int,
                                   nprobe: Optional[int] = None,
                                   warmup_queries: int = 10,
                                   repeat_times: int = 3) -> Dict[str, Any]:
        """
        Comprehensive search performance benchmark.
        
        Args:
            index: FAISS index object
            queries: Query vectors
            ground_truth: Ground truth indices
            k: Number of nearest neighbors
            nprobe: Number of probes for IVF indexes
            warmup_queries: Number of warmup queries
            repeat_times: Number of times to repeat the benchmark
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        n_queries = queries.shape[0]
        
        # Warmup
        if warmup_queries > 0:
            warmup_idx = min(warmup_queries, n_queries)
            self.logger.debug(f"Warming up with {warmup_idx} queries")
            index.search(queries[:warmup_idx], k, nprobe)
        
        # Benchmark
        search_times = []
        all_indices = []
        individual_latencies = []
        
        for run in range(repeat_times):
            self.logger.debug(f"Benchmark run {run + 1}/{repeat_times}")
            
            # Measure total search time
            start_time = time.time()
            distances, indices, search_time = index.search(queries, k, nprobe)
            total_time = time.time() - start_time
            
            search_times.append(search_time)
            all_indices.append(indices)
            
            # Measure individual query latencies (for a subset)
            if run == 0:  # Only measure individual latencies in first run
                sample_size = min(100, n_queries)  # Sample up to 100 queries
                sample_indices = np.random.choice(n_queries, sample_size, replace=False)
                
                for i in sample_indices:
                    query = queries[i:i+1]
                    start_time = time.time()
                    index.search(query, k, nprobe)
                    latency = time.time() - start_time
                    individual_latencies.append(latency)
        
        # Use results from the best run (fastest search time)
        best_run_idx = np.argmin(search_times)
        best_search_time = search_times[best_run_idx]
        best_indices = all_indices[best_run_idx]
        
        # Calculate metrics
        recall = self.calculate_recall(best_indices, ground_truth, k)
        precision = self.calculate_precision(best_indices, ground_truth, k)
        qps = self.calculate_qps(n_queries, best_search_time)
        
        # Latency statistics
        latency_stats = self.calculate_latency_stats(individual_latencies)
        
        # Memory usage
        memory_usage = self.get_memory_usage()
        gpu_memory = self.get_gpu_memory_usage()
        
        results = {
            'search_time': best_search_time,
            'search_time_mean': float(np.mean(search_times)),
            'search_time_std': float(np.std(search_times)),
            'qps': qps,
            'recall': recall,
            'precision': precision,
            'latency_stats': latency_stats,
            'memory_usage': memory_usage,
            'n_queries': n_queries,
            'k': k,
            'nprobe': nprobe,
            'repeat_times': repeat_times
        }
        
        if gpu_memory:
            results['gpu_memory'] = gpu_memory
        
        return results