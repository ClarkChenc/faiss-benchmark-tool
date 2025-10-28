"""
Results management for FAISS benchmark framework
"""

import json
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from ..utils.logger import get_logger


class BenchmarkResults:
    """Manager for benchmark results storage and analysis"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize results manager.
        
        Args:
            output_dir: Directory to store results
        """
        self.output_dir = output_dir
        self.logger = get_logger()
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def add_result(self, dataset_name: str, index_name: str, 
                  hardware_type: str, metrics: Dict[str, Any],
                  config: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a benchmark result.
        
        Args:
            dataset_name: Name of the dataset
            index_name: Name of the index
            hardware_type: Type of hardware (cpu/gpu)
            metrics: Performance metrics dictionary
            config: Additional configuration parameters
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'dataset_name': dataset_name,
            'index_name': index_name,
            'hardware_type': hardware_type,
            'metrics': metrics,
            'config': config or {}
        }
        
        self.results.append(result)
        self.logger.info(f"Added result: {dataset_name} - {index_name} - {hardware_type}")
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save results to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Saved {len(self.results)} results to {filepath}")
        return filepath
    
    def load_results(self, filepath: str) -> None:
        """
        Load results from JSON file.
        
        Args:
            filepath: Path to results file
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        self.logger.info(f"Loaded {len(self.results)} results from {filepath}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with flattened results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Flatten the results
        flattened_results = []
        
        for result in self.results:
            flat_result = {
                'timestamp': result['timestamp'],
                'dataset_name': result['dataset_name'],
                'index_name': result['index_name'],
                'hardware_type': result['hardware_type']
            }
            
            # Add metrics
            metrics = result['metrics']
            for key, value in metrics.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries (e.g., latency_stats)
                    for sub_key, sub_value in value.items():
                        flat_result[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_result[key] = value
            
            # Add config
            config = result.get('config', {})
            for key, value in config.items():
                flat_result[f"config_{key}"] = value
            
            flattened_results.append(flat_result)
        
        return pd.DataFrame(flattened_results)
    
    def save_csv(self, filename: Optional[str] = None) -> str:
        """
        Save results to CSV file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        df = self.to_dataframe()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Saved results to CSV: {filepath}")
        return filepath
    
    def get_comparison_table(self, metrics: List[str] = None,
                           group_by: List[str] = None) -> pd.DataFrame:
        """
        Get comparison table for different indexes/configurations.
        
        Args:
            metrics: List of metrics to include (default: key metrics)
            group_by: List of columns to group by
            
        Returns:
            Comparison DataFrame
        """
        df = self.to_dataframe()
        
        if df.empty:
            return df
        
        if metrics is None:
            metrics = ['search_time', 'qps', 'recall', 'precision', 'memory_usage_rss_mb']
        
        if group_by is None:
            group_by = ['dataset_name', 'index_name', 'hardware_type']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            self.logger.warning("No specified metrics found in results")
            return df
        
        # Group and aggregate
        comparison_cols = group_by + available_metrics
        available_cols = [col for col in comparison_cols if col in df.columns]
        
        if len(available_cols) < len(group_by):
            self.logger.warning("Some grouping columns not found in results")
            return df
        
        # Get the latest result for each group
        df_sorted = df.sort_values('timestamp')
        comparison_df = df_sorted.groupby(group_by)[available_metrics].last().reset_index()
        
        return comparison_df
    
    def get_best_performers(self, metric: str = 'qps', 
                           group_by: str = 'dataset_name',
                           ascending: bool = False) -> pd.DataFrame:
        """
        Get best performing indexes for each group.
        
        Args:
            metric: Metric to optimize for
            group_by: Column to group by
            ascending: Whether to sort in ascending order
            
        Returns:
            DataFrame with best performers
        """
        df = self.to_dataframe()
        
        if df.empty or metric not in df.columns:
            return df
        
        # Get best performer for each group
        best_performers = df.loc[df.groupby(group_by)[metric].idxmax() if not ascending 
                                else df.groupby(group_by)[metric].idxmin()]
        
        return best_performers
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of all results.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.to_dataframe()
        
        if df.empty:
            return {}
        
        summary = {
            'total_results': len(df),
            'datasets': df['dataset_name'].unique().tolist(),
            'indexes': df['index_name'].unique().tolist(),
            'hardware_types': df['hardware_type'].unique().tolist(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
        
        # Add metric statistics
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col in ['search_time', 'qps', 'recall', 'precision']:
                summary[f"{col}_stats"] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return summary
    
    def filter_results(self, **filters) -> 'BenchmarkResults':
        """
        Filter results based on criteria.
        
        Args:
            **filters: Filter criteria (e.g., dataset_name='sift1m')
            
        Returns:
            New BenchmarkResults object with filtered results
        """
        filtered_results = []
        
        for result in self.results:
            match = True
            
            for key, value in filters.items():
                if key in result and result[key] != value:
                    match = False
                    break
                elif key.startswith('metrics.'):
                    metric_key = key[8:]  # Remove 'metrics.' prefix
                    if metric_key in result['metrics'] and result['metrics'][metric_key] != value:
                        match = False
                        break
            
            if match:
                filtered_results.append(result)
        
        # Create new results object
        filtered_obj = BenchmarkResults(self.output_dir)
        filtered_obj.results = filtered_results
        
        return filtered_obj
    
    def clear_results(self) -> None:
        """Clear all results"""
        self.results.clear()
        self.logger.info("Cleared all results")
    
    def __len__(self) -> int:
        """Get number of results"""
        return len(self.results)
    
    def __str__(self) -> str:
        """String representation"""
        return f"BenchmarkResults({len(self.results)} results)"