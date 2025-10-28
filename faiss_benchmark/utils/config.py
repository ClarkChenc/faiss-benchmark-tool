"""
Configuration management for FAISS benchmark framework
"""

import yaml
import os
from typing import Dict, Any, List, Optional


class Config:
    """Configuration manager for FAISS benchmark framework"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'datasets.sift1m.dimension')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get all dataset configurations"""
        return self.get('datasets', {})
    
    def get_indexes(self) -> Dict[str, Dict[str, Any]]:
        """Get all index configurations"""
        return self.get('indexes', {})
    
    def get_benchmark_config(self) -> Dict[str, Any]:
        """Get benchmark configuration"""
        return self.get('benchmark', {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self.get('hardware', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.get('output', {})
    
    def get_k_values(self) -> List[int]:
        """Get k values for nearest neighbor search"""
        return self.get('benchmark.k_values', [1, 10, 100])
    
    def get_nprobe_values(self) -> List[int]:
        """Get nprobe values for IVF indexes"""
        return self.get('benchmark.nprobe_values', [1, 10, 100])
    
    def get_cpu_threads(self) -> List[int]:
        """Get CPU thread counts to test"""
        return self.get('benchmark.cpu_threads', [1, 4, 8, 16])
    
    def get_metrics(self) -> List[str]:
        """Get performance metrics to measure"""
        return self.get('benchmark.metrics', ['search_time', 'recall', 'qps'])
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU testing is enabled"""
        return self.get('hardware.gpu.enabled', False)
    
    def is_cpu_enabled(self) -> bool:
        """Check if CPU testing is enabled"""
        return self.get('hardware.cpu.enabled', True)
    
    def get_results_dir(self) -> str:
        """Get results directory path"""
        return self.get('output.results_dir', 'results')
    
    def get_plots_dir(self) -> str:
        """Get plots directory path"""
        return self.get('output.plots_dir', 'plots')
    
    def update(self, key: str, value: Any) -> None:
        """
        Update configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            output_path: Output file path (defaults to original config path)
        """
        path = output_path or self.config_path
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)