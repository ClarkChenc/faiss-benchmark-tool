"""
Dataset class for managing vector datasets in fvecs/ivecs format
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any
from ..utils.io_utils import read_fvecs, read_ivecs, get_file_info
from ..utils.logger import get_logger


class Dataset:
    """
    Dataset class for managing vector datasets in fvecs/ivecs format.
    
    Expected file naming convention:
    - {dataset_name}_base.fvecs: Base vectors for building index
    - {dataset_name}_query.fvecs: Query vectors for search
    - {dataset_name}_groundtruth.ivecs: Ground truth nearest neighbors
    """
    
    def __init__(self, name: str, data_dir: str, dimension: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            name: Dataset name
            data_dir: Directory containing dataset files
            dimension: Expected vector dimension (optional, will be auto-detected)
        """
        self.name = name
        self.data_dir = data_dir
        self.dimension = dimension
        self.logger = get_logger()
        
        # File paths
        self.base_file = os.path.join(data_dir, f"{name}_base.fvecs")
        self.query_file = os.path.join(data_dir, f"{name}_query.fvecs")
        self.groundtruth_file = os.path.join(data_dir, f"{name}_groundtruth.ivecs")
        
        # Cached data
        self._base_vectors = None
        self._query_vectors = None
        self._groundtruth = None
        
        # Dataset info
        self._info = self._get_dataset_info()
    
    @classmethod
    def _from_files(cls, name: str, base_file: str, query_file: str, 
                   groundtruth_file: str, dimension: Optional[int] = None):
        """
        Create dataset from specific file paths.
        
        Args:
            name: Dataset name
            base_file: Path to base vectors file
            query_file: Path to query vectors file
            groundtruth_file: Path to ground truth file
            dimension: Expected vector dimension (optional, will be auto-detected)
            
        Returns:
            Dataset object
        """
        # Create a dummy instance to set up the structure
        data_dir = os.path.dirname(base_file)
        instance = cls.__new__(cls)
        instance.name = name
        instance.data_dir = data_dir
        instance.dimension = dimension
        instance.logger = get_logger()
        
        # Set specific file paths
        instance.base_file = base_file
        instance.query_file = query_file
        instance.groundtruth_file = groundtruth_file
        
        # Cached data
        instance._base_vectors = None
        instance._query_vectors = None
        instance._groundtruth = None
        
        # Dataset info
        instance._info = instance._get_dataset_info()
        
        return instance
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        info = {
            "name": self.name,
            "data_dir": self.data_dir,
            "files_exist": {
                "base": os.path.exists(self.base_file),
                "query": os.path.exists(self.query_file),
                "groundtruth": os.path.exists(self.groundtruth_file)
            }
        }
        
        # Get file statistics
        if info["files_exist"]["base"]:
            try:
                n_base, dim_base = get_file_info(self.base_file)
                info["n_base"] = n_base
                info["dimension"] = dim_base
                if self.dimension is None:
                    self.dimension = dim_base
                elif self.dimension != dim_base:
                    self.logger.warning(f"Dimension mismatch: expected {self.dimension}, "
                                      f"got {dim_base} from {self.base_file}")
            except Exception as e:
                self.logger.error(f"Error reading base file {self.base_file}: {e}")
                info["n_base"] = 0
                info["dimension"] = 0
        
        if info["files_exist"]["query"]:
            try:
                n_query, dim_query = get_file_info(self.query_file)
                info["n_query"] = n_query
                if "dimension" in info and info["dimension"] != dim_query:
                    self.logger.warning(f"Query dimension mismatch: expected {info['dimension']}, "
                                      f"got {dim_query}")
            except Exception as e:
                self.logger.error(f"Error reading query file {self.query_file}: {e}")
                info["n_query"] = 0
        
        if info["files_exist"]["groundtruth"]:
            try:
                n_gt, k_gt = get_file_info(self.groundtruth_file)
                info["n_groundtruth"] = n_gt
                info["k_groundtruth"] = k_gt
            except Exception as e:
                self.logger.error(f"Error reading groundtruth file {self.groundtruth_file}: {e}")
                info["n_groundtruth"] = 0
                info["k_groundtruth"] = 0
        
        return info
    
    def is_valid(self) -> bool:
        """Check if dataset is valid (all required files exist)"""
        return all(self._info["files_exist"].values())
    
    def get_base_vectors(self) -> np.ndarray:
        """
        Get base vectors for index building.
        
        Returns:
            Base vectors array of shape (n_base, dimension)
        """
        if self._base_vectors is None:
            if not os.path.exists(self.base_file):
                raise FileNotFoundError(f"Base file not found: {self.base_file}")
            
            self.logger.info(f"Loading base vectors from {self.base_file}")
            self._base_vectors = read_fvecs(self.base_file)
            self.logger.info(f"Loaded {self._base_vectors.shape[0]} base vectors "
                           f"of dimension {self._base_vectors.shape[1]}")
        
        return self._base_vectors
    
    def get_query_vectors(self) -> np.ndarray:
        """
        Get query vectors for search.
        
        Returns:
            Query vectors array of shape (n_query, dimension)
        """
        if self._query_vectors is None:
            if not os.path.exists(self.query_file):
                raise FileNotFoundError(f"Query file not found: {self.query_file}")
            
            self.logger.info(f"Loading query vectors from {self.query_file}")
            self._query_vectors = read_fvecs(self.query_file)
            self.logger.info(f"Loaded {self._query_vectors.shape[0]} query vectors "
                           f"of dimension {self._query_vectors.shape[1]}")
        
        return self._query_vectors
    
    def get_groundtruth(self) -> np.ndarray:
        """
        Get ground truth nearest neighbors.
        
        Returns:
            Ground truth array of shape (n_query, k)
        """
        if self._groundtruth is None:
            if not os.path.exists(self.groundtruth_file):
                raise FileNotFoundError(f"Groundtruth file not found: {self.groundtruth_file}")
            
            self.logger.info(f"Loading ground truth from {self.groundtruth_file}")
            self._groundtruth = read_ivecs(self.groundtruth_file)
            self.logger.info(f"Loaded ground truth for {self._groundtruth.shape[0]} queries "
                           f"with k={self._groundtruth.shape[1]}")
        
        return self._groundtruth
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return self._info.copy()
    
    def clear_cache(self) -> None:
        """Clear cached data to free memory"""
        self._base_vectors = None
        self._query_vectors = None
        self._groundtruth = None
        self.logger.info(f"Cleared cache for dataset {self.name}")
    
    def __str__(self) -> str:
        """String representation"""
        info = self._info
        return (f"Dataset(name={self.name}, "
                f"n_base={info.get('n_base', 'unknown')}, "
                f"n_query={info.get('n_query', 'unknown')}, "
                f"dimension={info.get('dimension', 'unknown')}, "
                f"valid={self.is_valid()})")
    
    def __repr__(self) -> str:
        """String representation"""
        return self.__str__()