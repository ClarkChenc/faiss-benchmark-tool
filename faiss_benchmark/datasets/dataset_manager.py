"""
Dataset manager for handling multiple datasets
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
from .dataset import Dataset
from ..utils.config import Config
from ..utils.logger import get_logger


class DatasetManager:
    """Manager for handling multiple datasets"""
    
    def __init__(self, config: Config, data_root_dir: str = "data"):
        """
        Initialize dataset manager.
        
        Args:
            config: Configuration object
            data_root_dir: Root directory for dataset files
        """
        self.config = config
        self.data_root_dir = data_root_dir
        self.logger = get_logger()
        self._datasets = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_root_dir, exist_ok=True)
        
        # Load datasets from configuration
        self._load_datasets_from_config()
    
    def _load_datasets_from_config(self) -> None:
        """Load datasets from configuration"""
        dataset_configs = self.config.get_datasets()
        
        for dataset_name, dataset_config in dataset_configs.items():
            data_dir = os.path.join(self.data_root_dir, dataset_name)
            dimension = dataset_config.get('dimension')
            
            dataset = Dataset(dataset_name, data_dir, dimension)
            self._datasets[dataset_name] = dataset
            
            if dataset.is_valid():
                self.logger.info(f"Loaded dataset: {dataset}")
            else:
                self.logger.warning(f"Dataset {dataset_name} is not valid - "
                                  f"missing files in {data_dir}")
    
    def add_dataset(self, name: str, data_dir: Optional[str] = None,
                   base_file: Optional[str] = None,
                   query_file: Optional[str] = None,
                   ground_truth_file: Optional[str] = None,
                   dimension: Optional[int] = None) -> Dataset:
        """
        Add a new dataset.
        
        Args:
            name: Dataset name
            data_dir: Directory containing dataset files (if using standard naming)
            base_file: Path to base vectors file (alternative to data_dir)
            query_file: Path to query vectors file (alternative to data_dir)
            ground_truth_file: Path to ground truth file (alternative to data_dir)
            dimension: Expected vector dimension
            
        Returns:
            Dataset object
            
        Note:
            Either provide data_dir (for standard naming) or provide individual file paths.
        """
        if data_dir is not None:
            # Standard way: use data_dir with standard naming convention
            dataset = Dataset(name, data_dir, dimension)
        elif all([base_file, query_file, ground_truth_file]):
            # Alternative way: use specific file paths
            dataset = Dataset._from_files(name, base_file, query_file, ground_truth_file, dimension)
        else:
            raise ValueError("Either provide data_dir or all three file paths (base_file, query_file, ground_truth_file)")
        
        self._datasets[name] = dataset
        
        self.logger.info(f"Added dataset: {dataset}")
        return dataset
    
    def get_dataset(self, name: str) -> Dataset:
        """
        Get dataset by name.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset object
            
        Raises:
            KeyError: If dataset not found
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found. Available datasets: "
                          f"{list(self._datasets.keys())}")
        
        return self._datasets[name]
    
    def list_datasets(self) -> List[str]:
        """Get list of available dataset names"""
        return list(self._datasets.keys())
    
    def get_valid_datasets(self) -> List[str]:
        """Get list of valid dataset names (with all required files)"""
        return [name for name, dataset in self._datasets.items() 
                if dataset.is_valid()]
    
    def get_dataset_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get dataset information.
        
        Args:
            name: Dataset name (if None, returns info for all datasets)
            
        Returns:
            Dataset information dictionary
        """
        if name is not None:
            return self.get_dataset(name).get_info()
        
        return {name: dataset.get_info() 
                for name, dataset in self._datasets.items()}
    
    def validate_datasets(self) -> Dict[str, bool]:
        """
        Validate all datasets.
        
        Returns:
            Dictionary mapping dataset names to validation status
        """
        validation_results = {}
        
        for name, dataset in self._datasets.items():
            is_valid = dataset.is_valid()
            validation_results[name] = is_valid
            
            if is_valid:
                self.logger.info(f"Dataset {name}: VALID")
            else:
                missing_files = [file_type for file_type, exists 
                               in dataset.get_info()["files_exist"].items() 
                               if not exists]
                self.logger.warning(f"Dataset {name}: INVALID - "
                                  f"missing files: {missing_files}")
        
        return validation_results
    
    def create_random_dataset(self, name: str, n_base: int, n_query: int, 
                             dimension: int, k_groundtruth: int = 100,
                             seed: int = 42) -> Dataset:
        """
        Create a random dataset for testing.
        
        Args:
            name: Dataset name
            n_base: Number of base vectors
            n_query: Number of query vectors
            dimension: Vector dimension
            k_groundtruth: Number of ground truth neighbors
            seed: Random seed
            
        Returns:
            Created dataset
        """
        np.random.seed(seed)
        
        # Create dataset directory
        data_dir = os.path.join(self.data_root_dir, name)
        os.makedirs(data_dir, exist_ok=True)
        
        self.logger.info(f"Creating random dataset {name} with "
                        f"{n_base} base vectors, {n_query} queries, "
                        f"dimension {dimension}")
        
        # Generate random vectors
        base_vectors = np.random.randn(n_base, dimension).astype(np.float32)
        query_vectors = np.random.randn(n_query, dimension).astype(np.float32)
        
        # Compute ground truth using brute force
        self.logger.info("Computing ground truth...")
        distances = np.dot(query_vectors, base_vectors.T)
        groundtruth = np.argsort(-distances, axis=1)[:, :k_groundtruth].astype(np.int32)
        
        # Save to files
        from ..utils.io_utils import write_fvecs, write_ivecs
        
        base_file = os.path.join(data_dir, f"{name}_base.fvecs")
        query_file = os.path.join(data_dir, f"{name}_query.fvecs")
        gt_file = os.path.join(data_dir, f"{name}_groundtruth.ivecs")
        
        write_fvecs(base_file, base_vectors)
        write_fvecs(query_file, query_vectors)
        write_ivecs(gt_file, groundtruth)
        
        self.logger.info(f"Saved random dataset to {data_dir}")
        
        # Create and add dataset
        dataset = Dataset(name, data_dir, dimension)
        self._datasets[name] = dataset
        
        return dataset
    
    def clear_all_cache(self) -> None:
        """Clear cache for all datasets"""
        for dataset in self._datasets.values():
            dataset.clear_cache()
        self.logger.info("Cleared cache for all datasets")
    
    def get_summary(self) -> str:
        """Get summary of all datasets"""
        lines = ["Dataset Summary:"]
        lines.append("-" * 50)
        
        for name, dataset in self._datasets.items():
            info = dataset.get_info()
            status = "VALID" if dataset.is_valid() else "INVALID"
            lines.append(f"{name:15} | {status:7} | "
                        f"Base: {info.get('n_base', 'N/A'):8} | "
                        f"Query: {info.get('n_query', 'N/A'):6} | "
                        f"Dim: {info.get('dimension', 'N/A'):4}")
        
        return "\n".join(lines)