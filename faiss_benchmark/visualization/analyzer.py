"""
Results analysis module for FAISS benchmark framework
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..benchmarks.results import BenchmarkResults
from ..utils.logger import get_logger


class ResultsAnalyzer:
    """Analyzer for benchmark results with statistical analysis capabilities"""
    
    def __init__(self, results: BenchmarkResults):
        """
        Initialize results analyzer.
        
        Args:
            results: BenchmarkResults object
        """
        self.results = results
        self.logger = get_logger()
        self.df = results.to_dataframe()
    
    def analyze_performance_trends(self, metric: str = 'qps',
                                 group_by: str = 'index_name') -> Dict[str, Any]:
        """
        Analyze performance trends across different groups.
        
        Args:
            metric: Performance metric to analyze
            group_by: Column to group by
            
        Returns:
            Analysis results dictionary
        """
        if self.df.empty or metric not in self.df.columns:
            return {}
        
        analysis = {
            'metric': metric,
            'group_by': group_by,
            'groups': {}
        }
        
        for group_name, group_data in self.df.groupby(group_by):
            if metric in group_data.columns:
                values = group_data[metric].dropna()
                
                if len(values) > 0:
                    analysis['groups'][group_name] = {
                        'count': len(values),
                        'mean': float(values.mean()),
                        'std': float(values.std()) if len(values) > 1 else 0.0,
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                        'q25': float(values.quantile(0.25)),
                        'q75': float(values.quantile(0.75))
                    }
        
        return analysis
    
    def compare_algorithms(self, dataset_name: Optional[str] = None,
                          hardware_type: Optional[str] = None,
                          metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare different algorithms across specified metrics.
        
        Args:
            dataset_name: Filter by dataset name
            hardware_type: Filter by hardware type
            metrics: List of metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        df = self.df.copy()
        
        # Apply filters
        if dataset_name:
            df = df[df['dataset_name'] == dataset_name]
        
        if hardware_type:
            df = df[df['hardware_type'] == hardware_type]
        
        if df.empty:
            return pd.DataFrame()
        
        if metrics is None:
            metrics = ['qps', 'recall', 'precision', 'search_time', 'index_build_time']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            return df
        
        # Group by index and aggregate
        comparison = df.groupby('index_name')[available_metrics].agg(['mean', 'std', 'count']).round(4)
        
        return comparison
    
    def find_pareto_optimal(self, x_metric: str = 'search_time',
                           y_metric: str = 'recall',
                           minimize_x: bool = True,
                           maximize_y: bool = True) -> pd.DataFrame:
        """
        Find Pareto optimal solutions for two metrics.
        
        Args:
            x_metric: First metric (usually time/cost)
            y_metric: Second metric (usually quality)
            minimize_x: Whether to minimize x_metric
            maximize_y: Whether to maximize y_metric
            
        Returns:
            DataFrame with Pareto optimal solutions
        """
        df = self.df.copy()
        
        if df.empty or x_metric not in df.columns or y_metric not in df.columns:
            return pd.DataFrame()
        
        # Remove rows with missing values
        df = df.dropna(subset=[x_metric, y_metric])
        
        if df.empty:
            return df
        
        # Convert to numpy arrays for efficiency
        x_values = df[x_metric].values
        y_values = df[y_metric].values
        
        # Adjust for minimization/maximization
        if not minimize_x:
            x_values = -x_values
        if not maximize_y:
            y_values = -y_values
        
        # Find Pareto optimal points
        pareto_mask = np.zeros(len(df), dtype=bool)
        
        for i in range(len(df)):
            is_pareto = True
            for j in range(len(df)):
                if i != j:
                    # Check if point j dominates point i
                    if (x_values[j] <= x_values[i] and y_values[j] >= y_values[i] and
                        (x_values[j] < x_values[i] or y_values[j] > y_values[i])):
                        is_pareto = False
                        break
            pareto_mask[i] = is_pareto
        
        pareto_df = df[pareto_mask].copy()
        pareto_df = pareto_df.sort_values(x_metric)
        
        return pareto_df
    
    def analyze_hardware_efficiency(self) -> Dict[str, Any]:
        """
        Analyze efficiency differences between CPU and GPU.
        
        Returns:
            Hardware efficiency analysis
        """
        if 'hardware_type' not in self.df.columns:
            return {}
        
        cpu_data = self.df[self.df['hardware_type'] == 'cpu']
        gpu_data = self.df[self.df['hardware_type'] == 'gpu']
        
        analysis = {
            'cpu_results': len(cpu_data),
            'gpu_results': len(gpu_data),
            'comparisons': {}
        }
        
        # Compare metrics where both CPU and GPU data exist
        metrics = ['qps', 'search_time', 'recall', 'precision', 'memory_usage_rss_mb']
        
        for metric in metrics:
            if metric in self.df.columns:
                cpu_values = cpu_data[metric].dropna()
                gpu_values = gpu_data[metric].dropna()
                
                if len(cpu_values) > 0 and len(gpu_values) > 0:
                    analysis['comparisons'][metric] = {
                        'cpu_mean': float(cpu_values.mean()),
                        'gpu_mean': float(gpu_values.mean()),
                        'speedup': float(gpu_values.mean() / cpu_values.mean()) if cpu_values.mean() > 0 else None,
                        'cpu_std': float(cpu_values.std()) if len(cpu_values) > 1 else 0.0,
                        'gpu_std': float(gpu_values.std()) if len(gpu_values) > 1 else 0.0
                    }
        
        return analysis
    
    def analyze_scalability(self, size_metric: str = 'dataset_size',
                           performance_metric: str = 'qps') -> Dict[str, Any]:
        """
        Analyze scalability of different algorithms.
        
        Args:
            size_metric: Metric representing data size
            performance_metric: Performance metric to analyze
            
        Returns:
            Scalability analysis
        """
        if (size_metric not in self.df.columns or 
            performance_metric not in self.df.columns):
            return {}
        
        analysis = {
            'size_metric': size_metric,
            'performance_metric': performance_metric,
            'algorithms': {}
        }
        
        for algorithm, group_data in self.df.groupby('index_name'):
            if len(group_data) > 1:
                # Calculate correlation between size and performance
                size_values = group_data[size_metric].dropna()
                perf_values = group_data[performance_metric].dropna()
                
                if len(size_values) > 1 and len(perf_values) > 1:
                    # Align the data
                    common_indices = size_values.index.intersection(perf_values.index)
                    if len(common_indices) > 1:
                        aligned_size = size_values.loc[common_indices]
                        aligned_perf = perf_values.loc[common_indices]
                        
                        correlation = np.corrcoef(aligned_size, aligned_perf)[0, 1]
                        
                        analysis['algorithms'][algorithm] = {
                            'correlation': float(correlation) if not np.isnan(correlation) else None,
                            'data_points': len(common_indices),
                            'size_range': {
                                'min': float(aligned_size.min()),
                                'max': float(aligned_size.max())
                            },
                            'performance_range': {
                                'min': float(aligned_perf.min()),
                                'max': float(aligned_perf.max())
                            }
                        }
        
        return analysis
    
    def generate_ranking(self, metrics: List[str] = None,
                        weights: List[float] = None,
                        higher_is_better: List[bool] = None) -> pd.DataFrame:
        """
        Generate overall ranking of algorithms based on multiple metrics.
        
        Args:
            metrics: List of metrics to consider
            weights: Weights for each metric (equal if None)
            higher_is_better: Whether higher values are better for each metric
            
        Returns:
            DataFrame with rankings
        """
        if metrics is None:
            metrics = ['qps', 'recall']
        
        if weights is None:
            weights = [1.0] * len(metrics)
        
        if higher_is_better is None:
            higher_is_better = [True] * len(metrics)
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        if not available_metrics:
            return pd.DataFrame()
        
        # Adjust weights and higher_is_better for available metrics
        available_weights = [weights[i] for i, m in enumerate(metrics) if m in available_metrics]
        available_higher_is_better = [higher_is_better[i] for i, m in enumerate(metrics) if m in available_metrics]
        
        # Group by algorithm and calculate mean values
        algorithm_means = self.df.groupby('index_name')[available_metrics].mean()
        
        # Normalize metrics to 0-1 scale
        normalized = algorithm_means.copy()
        for i, metric in enumerate(available_metrics):
            values = algorithm_means[metric].dropna()
            if len(values) > 0:
                min_val = values.min()
                max_val = values.max()
                
                if max_val > min_val:
                    if available_higher_is_better[i]:
                        normalized[metric] = (values - min_val) / (max_val - min_val)
                    else:
                        normalized[metric] = (max_val - values) / (max_val - min_val)
                else:
                    normalized[metric] = 1.0
        
        # Calculate weighted score
        normalized['weighted_score'] = 0.0
        total_weight = sum(available_weights)
        
        for i, metric in enumerate(available_metrics):
            weight = available_weights[i] / total_weight
            normalized['weighted_score'] += normalized[metric] * weight
        
        # Sort by score
        ranking = normalized.sort_values('weighted_score', ascending=False)
        ranking['rank'] = range(1, len(ranking) + 1)
        
        return ranking
    
    def detect_anomalies(self, metric: str = 'qps',
                        method: str = 'iqr',
                        threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect anomalous results in the data.
        
        Args:
            metric: Metric to analyze for anomalies
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomalous results
        """
        if metric not in self.df.columns:
            return pd.DataFrame()
        
        values = self.df[metric].dropna()
        
        if len(values) == 0:
            return pd.DataFrame()
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            anomaly_mask = (values < lower_bound) | (values > upper_bound)
            
        elif method == 'zscore':
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val > 0:
                z_scores = np.abs((values - mean_val) / std_val)
                anomaly_mask = z_scores > threshold
            else:
                anomaly_mask = pd.Series([False] * len(values), index=values.index)
        
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        anomalies = self.df.loc[anomaly_mask.index[anomaly_mask]]
        
        return anomalies
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Returns:
            Summary report dictionary
        """
        report = {
            'overview': {
                'total_results': len(self.df),
                'datasets': self.df['dataset_name'].nunique() if 'dataset_name' in self.df.columns else 0,
                'algorithms': self.df['index_name'].nunique() if 'index_name' in self.df.columns else 0,
                'hardware_types': self.df['hardware_type'].nunique() if 'hardware_type' in self.df.columns else 0
            }
        }
        
        # Performance trends
        if 'qps' in self.df.columns:
            report['qps_analysis'] = self.analyze_performance_trends('qps', 'index_name')
        
        if 'recall' in self.df.columns:
            report['recall_analysis'] = self.analyze_performance_trends('recall', 'index_name')
        
        # Hardware efficiency
        report['hardware_efficiency'] = self.analyze_hardware_efficiency()
        
        # Best performers
        if 'qps' in self.df.columns:
            best_qps = self.df.loc[self.df['qps'].idxmax()] if not self.df['qps'].isna().all() else None
            if best_qps is not None:
                report['best_qps'] = {
                    'algorithm': best_qps.get('index_name', 'Unknown'),
                    'dataset': best_qps.get('dataset_name', 'Unknown'),
                    'value': float(best_qps['qps'])
                }
        
        if 'recall' in self.df.columns:
            best_recall = self.df.loc[self.df['recall'].idxmax()] if not self.df['recall'].isna().all() else None
            if best_recall is not None:
                report['best_recall'] = {
                    'algorithm': best_recall.get('index_name', 'Unknown'),
                    'dataset': best_recall.get('dataset_name', 'Unknown'),
                    'value': float(best_recall['recall'])
                }
        
        return report