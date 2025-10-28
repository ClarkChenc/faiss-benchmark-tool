"""
FAISS index wrapper for benchmark framework
"""

import time
import numpy as np
import faiss
from typing import Dict, Any, Optional, Tuple, List
from ..utils.logger import get_logger


class FaissIndex:
    """Wrapper for FAISS index with benchmarking capabilities"""
    
    def __init__(self, name: str, factory_string: str, dimension: int,
                 use_gpu: bool = False, gpu_id: int = 0):
        """
        Initialize FAISS index.
        
        Args:
            name: Index name
            factory_string: FAISS factory string (e.g., "IVF100,Flat")
            dimension: Vector dimension
            use_gpu: Whether to use GPU
            gpu_id: GPU device ID
        """
        self.name = name
        self.factory_string = factory_string
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.logger = get_logger()
        
        # Index objects
        self.index = None
        self.gpu_resource = None
        
        # Performance metrics
        self.build_time = 0.0
        self.memory_usage = 0.0
        self.is_trained = False
        self.n_vectors = 0
        
        # Create index
        self._create_index()
    
    def _create_index(self) -> None:
        """Create FAISS index from factory string"""
        try:
            # Create CPU index
            self.index = faiss.index_factory(self.dimension, self.factory_string)
            
            # Move to GPU if requested
            if self.use_gpu:
                self._setup_gpu()
            
            self.logger.info(f"Created {self.name} index: {self.factory_string}")
            
        except Exception as e:
            self.logger.error(f"Failed to create index {self.name}: {e}")
            raise
    
    def _setup_gpu(self) -> None:
        """Setup GPU resources"""
        try:
            # Check if GPU is available
            if not faiss.get_num_gpus():
                raise RuntimeError("No GPU available")
            
            # Create GPU resource
            self.gpu_resource = faiss.StandardGpuResources()
            
            # Move index to GPU
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_resource, self.gpu_id, self.index
            )
            
            self.logger.info(f"Moved {self.name} index to GPU {self.gpu_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup GPU for {self.name}: {e}")
            self.use_gpu = False
            raise
    
    def train(self, vectors: np.ndarray) -> float:
        """
        Train the index if needed.
        
        Args:
            vectors: Training vectors
            
        Returns:
            Training time in seconds
        """
        if self.index.is_trained:
            self.logger.info(f"Index {self.name} is already trained")
            self.is_trained = True
            return 0.0
        
        self.logger.info(f"Training {self.name} index with {vectors.shape[0]} vectors")
        
        start_time = time.time()
        self.index.train(vectors)
        train_time = time.time() - start_time
        
        self.is_trained = True
        self.logger.info(f"Trained {self.name} index in {train_time:.2f} seconds")
        
        return train_time
    
    def add_vectors(self, vectors: np.ndarray) -> float:
        """
        Add vectors to the index.
        
        Args:
            vectors: Vectors to add
            
        Returns:
            Time taken to add vectors in seconds
        """
        if not self.index.is_trained:
            raise RuntimeError(f"Index {self.name} must be trained before adding vectors")
        
        self.logger.info(f"Adding {vectors.shape[0]} vectors to {self.name} index")
        
        start_time = time.time()
        self.index.add(vectors)
        add_time = time.time() - start_time
        
        self.n_vectors = self.index.ntotal
        self.logger.info(f"Added vectors to {self.name} index in {add_time:.2f} seconds")
        
        return add_time
    
    def build(self, vectors: np.ndarray) -> float:
        """
        Build the index (train + add).
        
        Args:
            vectors: Vectors to build index with
            
        Returns:
            Total build time in seconds
        """
        start_time = time.time()
        
        # Train if needed
        train_time = self.train(vectors)
        
        # Add vectors
        add_time = self.add_vectors(vectors)
        
        self.build_time = time.time() - start_time
        
        self.logger.info(f"Built {self.name} index in {self.build_time:.2f} seconds "
                        f"(train: {train_time:.2f}s, add: {add_time:.2f}s)")
        
        return self.build_time
    
    def search(self, queries: np.ndarray, k: int, 
               nprobe: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Search for nearest neighbors.
        
        Args:
            queries: Query vectors
            k: Number of nearest neighbors
            nprobe: Number of probes for IVF indexes
            
        Returns:
            Tuple of (distances, indices, search_time)
        """
        if self.index.ntotal == 0:
            raise RuntimeError(f"Index {self.name} is empty")
        
        # Set nprobe for IVF indexes
        if nprobe is not None and hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        start_time = time.time()
        distances, indices = self.index.search(queries, k)
        search_time = time.time() - start_time
        
        return distances, indices, search_time
    
    def get_memory_usage(self) -> float:
        """
        Get memory usage of the index in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            if self.use_gpu:
                # For GPU indexes, estimate based on index size
                # This is an approximation as FAISS doesn't provide exact GPU memory usage
                memory_bytes = self.index.ntotal * self.dimension * 4  # float32
                if hasattr(self.index, 'code_size'):
                    memory_bytes = self.index.ntotal * self.index.code_size
            else:
                # For CPU indexes, we can get more accurate memory usage
                memory_bytes = 0
                if hasattr(self.index, 'sa_code_size'):
                    memory_bytes += self.index.ntotal * self.index.sa_code_size()
                else:
                    memory_bytes = self.index.ntotal * self.dimension * 4  # float32
            
            self.memory_usage = memory_bytes / (1024 * 1024)  # Convert to MB
            return self.memory_usage
            
        except Exception as e:
            self.logger.warning(f"Could not get memory usage for {self.name}: {e}")
            return 0.0
    
    def set_search_params(self, params: Dict[str, Any]) -> None:
        """
        Set search parameters.
        
        Args:
            params: Dictionary of search parameters
        """
        for param, value in params.items():
            if hasattr(self.index, param):
                setattr(self.index, param, value)
                self.logger.debug(f"Set {param}={value} for {self.name}")
            else:
                self.logger.warning(f"Parameter {param} not available for {self.name}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get index information"""
        return {
            "name": self.name,
            "factory_string": self.factory_string,
            "dimension": self.dimension,
            "use_gpu": self.use_gpu,
            "gpu_id": self.gpu_id if self.use_gpu else None,
            "is_trained": self.is_trained,
            "n_vectors": self.n_vectors,
            "build_time": self.build_time,
            "memory_usage_mb": self.get_memory_usage(),
            "index_type": type(self.index).__name__
        }
    
    def save(self, filepath: str) -> None:
        """
        Save index to file.
        
        Args:
            filepath: Output file path
        """
        try:
            if self.use_gpu:
                # Move to CPU before saving
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, filepath)
            else:
                faiss.write_index(self.index, filepath)
            
            self.logger.info(f"Saved {self.name} index to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save {self.name} index: {e}")
            raise
    
    def load(self, filepath: str) -> None:
        """
        Load index from file.
        
        Args:
            filepath: Input file path
        """
        try:
            # Load CPU index
            cpu_index = faiss.read_index(filepath)
            
            if self.use_gpu:
                # Move to GPU
                self.index = faiss.index_cpu_to_gpu(
                    self.gpu_resource, self.gpu_id, cpu_index
                )
            else:
                self.index = cpu_index
            
            self.n_vectors = self.index.ntotal
            self.is_trained = self.index.is_trained
            
            self.logger.info(f"Loaded {self.name} index from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load {self.name} index: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation"""
        return (f"FaissIndex(name={self.name}, "
                f"factory={self.factory_string}, "
                f"dim={self.dimension}, "
                f"gpu={self.use_gpu}, "
                f"n_vectors={self.n_vectors})")
    
    def __repr__(self) -> str:
        """String representation"""
        return self.__str__()