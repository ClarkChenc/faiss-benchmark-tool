"""
Index manager for handling multiple FAISS indexes
"""

import os
from typing import Dict, List, Any, Optional, Iterator
from .faiss_index import FaissIndex
from ..utils.config import Config
from ..utils.logger import get_logger


class IndexManager:
    """Manager for handling multiple FAISS indexes"""
    
    def __init__(self, config: Config):
        """
        Initialize index manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger()
        self._indexes = {}
        
        # Load index configurations
        self._load_indexes_from_config()
    
    def _get_factory_string(self, index_type: str, **params) -> str:
        """
        Convert index type and parameters to FAISS factory string.
        
        Args:
            index_type: Index type (e.g., "Flat", "IVFFlat", "IVFPQ")
            **params: Additional index parameters
            
        Returns:
            FAISS factory string
        """
        if index_type == "Flat":
            return "Flat"
        elif index_type == "IVFFlat":
            nlist = params.get('nlist', 100)
            return f"IVF{nlist},Flat"
        elif index_type == "IVFPQ":
            nlist = params.get('nlist', 100)
            m = params.get('m', 8)
            nbits = params.get('nbits', 8)
            return f"IVF{nlist},PQ{m}x{nbits}"
        elif index_type == "PQ":
            m = params.get('m', 8)
            nbits = params.get('nbits', 8)
            return f"PQ{m}x{nbits}"
        elif index_type == "LSH":
            nbits = params.get('nbits', 64)
            return f"LSH{nbits}"
        elif index_type == "HNSW":
            m = params.get('m', 16)
            return f"HNSW{m}"
        else:
            # Assume it's already a factory string
            return index_type
    
    def _load_indexes_from_config(self) -> None:
        """Load index configurations from config"""
        index_configs = self.config.get_indexes()
        
        for index_name, index_config in index_configs.items():
            self.logger.info(f"Loaded index configuration: {index_name}")
            # Note: Actual index creation is deferred until needed
            # to avoid creating indexes for all parameter combinations upfront
    
    def create_index(self, index_type: str, dimension: int,
                    use_gpu: bool = False, gpu_id: int = 0, **params) -> FaissIndex:
        """
        Create a new FAISS index.
        
        Args:
            index_type: Index type (e.g., "Flat", "IVFFlat", "IVFPQ")
            dimension: Vector dimension
            use_gpu: Whether to use GPU
            gpu_id: GPU device ID
            **params: Additional index parameters
            
        Returns:
            FaissIndex object
        """
        # Convert index type to factory string
        factory_string = self._get_factory_string(index_type, **params)
        
        # Generate unique name for this index instance
        name = f"{index_type}_{len(self._indexes)}"
        
        index = FaissIndex(name, factory_string, dimension, use_gpu, gpu_id)
        self._indexes[name] = index
        
        self.logger.info(f"Created index: {index}")
        return index
    
    def get_index(self, name: str) -> FaissIndex:
        """
        Get index by name.
        
        Args:
            name: Index name
            
        Returns:
            FaissIndex object
            
        Raises:
            KeyError: If index not found
        """
        if name not in self._indexes:
            raise KeyError(f"Index '{name}' not found. Available indexes: "
                          f"{list(self._indexes.keys())}")
        
        return self._indexes[name]
    
    def list_indexes(self) -> List[str]:
        """Get list of created index names"""
        return list(self._indexes.keys())
    
    def get_index_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all index configurations from config"""
        return self.config.get_indexes()
    
    def create_indexes_for_dataset(self, dimension: int, 
                                  use_gpu: bool = False) -> List[FaissIndex]:
        """
        Create all configured indexes for a given dataset dimension.
        
        Args:
            dimension: Vector dimension
            use_gpu: Whether to use GPU
            
        Returns:
            List of created FaissIndex objects
        """
        created_indexes = []
        index_configs = self.config.get_indexes()
        
        for index_name, index_config in index_configs.items():
            # Check GPU compatibility
            if use_gpu and not index_config.get('gpu_compatible', True):
                self.logger.warning(f"Skipping {index_name} - not GPU compatible")
                continue
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(index_config)
            
            for i, params in enumerate(param_combinations):
                # Create unique name for this parameter combination
                if len(param_combinations) > 1:
                    unique_name = f"{index_name}_{i}"
                else:
                    unique_name = index_name
                
                # Format factory string with parameters
                factory_string = index_config['factory_string'].format(**params)
                
                try:
                    index = self.create_index(
                        unique_name, factory_string, dimension, use_gpu
                    )
                    created_indexes.append(index)
                    
                except Exception as e:
                    self.logger.error(f"Failed to create index {unique_name}: {e}")
        
        return created_indexes
    
    def _generate_param_combinations(self, index_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for an index configuration.
        
        Args:
            index_config: Index configuration dictionary
            
        Returns:
            List of parameter dictionaries
        """
        import itertools
        
        # Extract parameter lists
        param_lists = {}
        for key, value in index_config.items():
            if key not in ['name', 'factory_string', 'gpu_compatible']:
                if isinstance(value, list):
                    param_lists[key] = value
                else:
                    param_lists[key] = [value]
        
        if not param_lists:
            return [{}]
        
        # Generate all combinations
        keys = list(param_lists.keys())
        value_combinations = itertools.product(*[param_lists[key] for key in keys])
        
        combinations = []
        for values in value_combinations:
            combination = dict(zip(keys, values))
            combinations.append(combination)
        
        return combinations
    
    def benchmark_index_creation(self, index_configs: List[Dict[str, Any]], 
                                dimension: int, use_gpu: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark index creation for multiple configurations.
        
        Args:
            index_configs: List of index configuration dictionaries
            dimension: Vector dimension
            use_gpu: Whether to use GPU
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for config in index_configs:
            name = config['name']
            factory_string = config['factory_string']
            
            try:
                self.logger.info(f"Benchmarking index creation: {name}")
                
                import time
                start_time = time.time()
                
                index = self.create_index(name, factory_string, dimension, use_gpu)
                
                creation_time = time.time() - start_time
                
                results[name] = {
                    'creation_time': creation_time,
                    'success': True,
                    'info': index.get_info()
                }
                
                self.logger.info(f"Created {name} in {creation_time:.4f} seconds")
                
            except Exception as e:
                results[name] = {
                    'creation_time': 0.0,
                    'success': False,
                    'error': str(e)
                }
                self.logger.error(f"Failed to create {name}: {e}")
        
        return results
    
    def save_all_indexes(self, output_dir: str) -> None:
        """
        Save all indexes to files.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for name, index in self._indexes.items():
            if index.n_vectors > 0:  # Only save non-empty indexes
                filepath = os.path.join(output_dir, f"{name}.index")
                try:
                    index.save(filepath)
                except Exception as e:
                    self.logger.error(f"Failed to save index {name}: {e}")
    
    def load_index(self, name: str, filepath: str, dimension: int,
                  use_gpu: bool = False) -> FaissIndex:
        """
        Load index from file.
        
        Args:
            name: Index name
            filepath: Input file path
            dimension: Vector dimension
            use_gpu: Whether to use GPU
            
        Returns:
            Loaded FaissIndex object
        """
        # Create empty index first
        index = FaissIndex(name, "Flat", dimension, use_gpu)
        
        # Load from file
        index.load(filepath)
        
        # Store in manager
        self._indexes[name] = index
        
        return index
    
    def clear_all_indexes(self) -> None:
        """Clear all indexes to free memory"""
        self._indexes.clear()
        self.logger.info("Cleared all indexes")
    
    def get_summary(self) -> str:
        """Get summary of all indexes"""
        if not self._indexes:
            return "No indexes created"
        
        lines = ["Index Summary:"]
        lines.append("-" * 80)
        lines.append(f"{'Name':<20} {'Type':<15} {'Vectors':<10} {'GPU':<5} {'Memory(MB)':<12}")
        lines.append("-" * 80)
        
        for name, index in self._indexes.items():
            info = index.get_info()
            lines.append(f"{name:<20} {info['factory_string']:<15} "
                        f"{info['n_vectors']:<10} {info['use_gpu']!s:<5} "
                        f"{info['memory_usage_mb']:<12.2f}")
        
        return "\n".join(lines)
    
    def __iter__(self) -> Iterator[FaissIndex]:
        """Iterate over indexes"""
        return iter(self._indexes.values())
    
    def __len__(self) -> int:
        """Get number of indexes"""
        return len(self._indexes)