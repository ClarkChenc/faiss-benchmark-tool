"""
Main benchmark runner for FAISS performance testing
"""

import time
import gc
import psutil
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ..datasets.dataset_manager import DatasetManager
from ..indexes.index_manager import IndexManager
from ..utils.config import Config
from ..utils.logger import get_logger
from .metrics import MetricsCalculator
from .results import BenchmarkResults


class BenchmarkRunner:
    """Main class for running FAISS benchmarks"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize benchmark runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.logger = get_logger()
        
        # Initialize managers
        self.dataset_manager = DatasetManager(self.config)
        self.index_manager = IndexManager(self.config)
        self.metrics_calculator = MetricsCalculator()
        self.results = BenchmarkResults(self.config.get('output.results_dir', 'results'))
        
        # Benchmark settings
        self.warmup_queries = self.config.get('benchmark.warmup_queries', 100)
        self.test_queries = self.config.get('benchmark.test_queries', 1000)
        self.cpu_threads = self.config.get('hardware.cpu_threads', 4)
        
        self.logger.info("BenchmarkRunner initialized")
    
    def run_single_benchmark(self, dataset_name: str, index_config: Dict[str, Any],
                           hardware_type: str = 'cpu', k: int = 10) -> Dict[str, Any]:
        """
        Run benchmark for a single dataset-index-hardware combination.
        
        Args:
            dataset_name: Name of the dataset
            index_config: Index configuration
            hardware_type: 'cpu' or 'gpu'
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Dictionary with benchmark results
        """
        self.logger.benchmark_start(f"{dataset_name} - {index_config['type']} - {hardware_type}")
        
        try:
            # Load dataset
            dataset = self.dataset_manager.get_dataset(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_name} not found")
            
            # Get data
            base_vectors = dataset.get_base_vectors()
            query_vectors = dataset.get_query_vectors()
            ground_truth = dataset.get_ground_truth()
            
            if query_vectors is None:
                raise ValueError(f"No query vectors found for dataset {dataset_name}")
            
            # Limit test queries
            if len(query_vectors) > self.test_queries:
                query_indices = np.random.choice(len(query_vectors), self.test_queries, replace=False)
                query_vectors = query_vectors[query_indices]
                if ground_truth is not None:
                    ground_truth = ground_truth[query_indices]
            
            # Create and build index
            use_gpu = (hardware_type == 'gpu')
            index = self.index_manager.create_index(
                index_config['type'], 
                base_vectors.shape[1],
                use_gpu=use_gpu,
                **index_config.get('params', {})
            )
            
            # Set CPU threads for CPU benchmarks
            if hardware_type == 'cpu':
                import faiss
                faiss.omp_set_num_threads(self.cpu_threads)
            
            # Measure index build time
            build_start = time.time()
            index.build(base_vectors)
            build_time = time.time() - build_start
            
            self.logger.index_built(index_config['type'], build_time, len(base_vectors))
            
            # Warmup
            if self.warmup_queries > 0:
                warmup_queries = query_vectors[:min(self.warmup_queries, len(query_vectors))]
                _, _ = index.search(warmup_queries, k)
                gc.collect()  # Clean up after warmup
            
            # Run benchmark
            search_results = self.metrics_calculator.benchmark_search_performance(
                index, query_vectors, k, ground_truth
            )
            
            # Get memory usage
            memory_usage = self.metrics_calculator.get_memory_usage()
            
            # Get GPU memory if using GPU
            gpu_memory = None
            if use_gpu:
                gpu_memory = self.metrics_calculator.get_gpu_memory_usage()
            
            # Compile results
            results = {
                'dataset_name': dataset_name,
                'index_type': index_config['type'],
                'hardware_type': hardware_type,
                'index_build_time': build_time,
                'dataset_size': len(base_vectors),
                'query_size': len(query_vectors),
                'dimension': base_vectors.shape[1],
                'k': k,
                'cpu_threads': self.cpu_threads if hardware_type == 'cpu' else None,
                **search_results,
                **memory_usage
            }
            
            if gpu_memory:
                results.update(gpu_memory)
            
            # Add index parameters
            results['index_params'] = index_config.get('params', {})
            
            self.logger.benchmark_end(f"{dataset_name} - {index_config['type']} - {hardware_type}")
            self.logger.search_performance(
                results['qps'], 
                results['search_time'], 
                results.get('recall', 0)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {str(e)}")
            raise
    
    def run_dataset_benchmark(self, dataset_name: str, 
                            index_configs: Optional[List[Dict[str, Any]]] = None,
                            hardware_types: Optional[List[str]] = None,
                            k_values: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Run benchmarks for a dataset across multiple configurations.
        
        Args:
            dataset_name: Name of the dataset
            index_configs: List of index configurations (from config if None)
            hardware_types: List of hardware types (from config if None)
            k_values: List of k values (from config if None)
            
        Returns:
            List of benchmark results
        """
        if index_configs is None:
            index_configs = self.config.get('indexes', [])
        
        if hardware_types is None:
            hardware_types = self.config.get('benchmark.hardware_types', ['cpu'])
        
        if k_values is None:
            k_values = self.config.get('benchmark.k_values', [10])
        
        all_results = []
        
        for index_config in index_configs:
            for hardware_type in hardware_types:
                for k in k_values:
                    try:
                        result = self.run_single_benchmark(
                            dataset_name, index_config, hardware_type, k
                        )
                        all_results.append(result)
                        
                        # Add to results manager
                        self.results.add_result(
                            dataset_name=dataset_name,
                            index_name=index_config['type'],
                            hardware_type=hardware_type,
                            metrics=result,
                            config={'k': k, **index_config.get('params', {})}
                        )
                        
                    except Exception as e:
                        self.logger.error(
                            f"Failed benchmark: {dataset_name} - {index_config['type']} - "
                            f"{hardware_type} - k={k}: {str(e)}"
                        )
                        continue
        
        return all_results
    
    def run_full_benchmark(self, dataset_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run full benchmark across all datasets and configurations.
        
        Args:
            dataset_names: List of dataset names (all available if None)
            
        Returns:
            List of all benchmark results
        """
        if dataset_names is None:
            dataset_names = self.dataset_manager.list_datasets()
        
        all_results = []
        
        for dataset_name in dataset_names:
            self.logger.info(f"Starting benchmark for dataset: {dataset_name}")
            
            try:
                dataset_results = self.run_dataset_benchmark(dataset_name)
                all_results.extend(dataset_results)
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark dataset {dataset_name}: {str(e)}")
                continue
        
        # Save results
        results_file = self.results.save_results()
        csv_file = self.results.save_csv()
        
        self.logger.info(f"Benchmark completed. Results saved to {results_file} and {csv_file}")
        
        return all_results
    
    def run_comparison_benchmark(self, dataset_name: str, 
                               index_types: List[str],
                               hardware_type: str = 'cpu',
                               k: int = 10) -> Dict[str, Any]:
        """
        Run comparison benchmark for specific index types on a dataset.
        
        Args:
            dataset_name: Name of the dataset
            index_types: List of index types to compare
            hardware_type: Hardware type to use
            k: Number of nearest neighbors
            
        Returns:
            Comparison results
        """
        results = []
        
        # Get index configurations for specified types
        all_index_configs = self.config.get('indexes', [])
        selected_configs = [
            config for config in all_index_configs 
            if config['type'] in index_types
        ]
        
        if not selected_configs:
            raise ValueError(f"No configurations found for index types: {index_types}")
        
        # Run benchmarks
        for index_config in selected_configs:
            try:
                result = self.run_single_benchmark(
                    dataset_name, index_config, hardware_type, k
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(
                    f"Failed comparison benchmark: {index_config['type']}: {str(e)}"
                )
                continue
        
        # Create comparison summary
        comparison = {
            'dataset_name': dataset_name,
            'hardware_type': hardware_type,
            'k': k,
            'results': results,
            'summary': self._create_comparison_summary(results)
        }
        
        return comparison
    
    def _create_comparison_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary of comparison results.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Summary dictionary
        """
        if not results:
            return {}
        
        summary = {
            'best_qps': None,
            'best_recall': None,
            'fastest_build': None,
            'lowest_memory': None
        }
        
        # Find best performers
        best_qps = max(results, key=lambda x: x.get('qps', 0))
        summary['best_qps'] = {
            'index_type': best_qps['index_type'],
            'qps': best_qps.get('qps', 0)
        }
        
        if any('recall' in r for r in results):
            best_recall = max(results, key=lambda x: x.get('recall', 0))
            summary['best_recall'] = {
                'index_type': best_recall['index_type'],
                'recall': best_recall.get('recall', 0)
            }
        
        fastest_build = min(results, key=lambda x: x.get('index_build_time', float('inf')))
        summary['fastest_build'] = {
            'index_type': fastest_build['index_type'],
            'build_time': fastest_build.get('index_build_time', 0)
        }
        
        lowest_memory = min(results, key=lambda x: x.get('memory_usage_rss_mb', float('inf')))
        summary['lowest_memory'] = {
            'index_type': lowest_memory['index_type'],
            'memory_mb': lowest_memory.get('memory_usage_rss_mb', 0)
        }
        
        return summary
    
    def get_results_manager(self) -> BenchmarkResults:
        """Get the results manager for further analysis"""
        return self.results
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.dataset_manager.clear_cache()
        self.index_manager.clear_indexes()
        self.results.clear_results()
        gc.collect()
        self.logger.info("Cleared all caches")