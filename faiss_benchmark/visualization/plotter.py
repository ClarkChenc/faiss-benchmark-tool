"""
Plotting and visualization module for FAISS benchmark results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, Any, List, Optional, Tuple

from ..benchmarks.results import BenchmarkResults
from ..utils.logger import get_logger


class BenchmarkPlotter:
    """Plotter for benchmark results with various visualization options"""
    
    def __init__(self, results: BenchmarkResults, output_dir: str = "plots"):
        """
        Initialize benchmark plotter.
        
        Args:
            results: BenchmarkResults object
            output_dir: Directory to save plots
        """
        self.results = results
        self.output_dir = output_dir
        self.logger = get_logger()
        self.df = results.to_dataframe()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_performance_comparison(self, metric: str = 'qps',
                                  group_by: str = 'index_name',
                                  filter_by: Optional[Dict[str, Any]] = None,
                                  save_path: Optional[str] = None,
                                  interactive: bool = False) -> str:
        """
        Plot performance comparison across different groups.
        
        Args:
            metric: Performance metric to plot
            group_by: Column to group by
            filter_by: Dictionary of filters to apply
            save_path: Path to save the plot
            interactive: Whether to create interactive plot
            
        Returns:
            Path to saved plot
        """
        df = self.df.copy()
        
        # Apply filters
        if filter_by:
            for key, value in filter_by.items():
                if key in df.columns:
                    df = df[df[key] == value]
        
        if df.empty or metric not in df.columns:
            self.logger.warning(f"No data available for metric: {metric}")
            return ""
        
        if interactive:
            return self._plot_interactive_comparison(df, metric, group_by, save_path)
        else:
            return self._plot_static_comparison(df, metric, group_by, save_path)
    
    def _plot_static_comparison(self, df: pd.DataFrame, metric: str,
                              group_by: str, save_path: Optional[str]) -> str:
        """Create static comparison plot"""
        plt.figure(figsize=(12, 8))
        
        # Box plot
        sns.boxplot(data=df, x=group_by, y=metric)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{metric.upper()} Comparison by {group_by.replace("_", " ").title()}')
        plt.ylabel(metric.upper())
        plt.xlabel(group_by.replace("_", " ").title())
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{metric}_comparison_{group_by}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved comparison plot: {save_path}")
        return save_path
    
    def _plot_interactive_comparison(self, df: pd.DataFrame, metric: str,
                                   group_by: str, save_path: Optional[str]) -> str:
        """Create interactive comparison plot"""
        fig = px.box(df, x=group_by, y=metric,
                    title=f'{metric.upper()} Comparison by {group_by.replace("_", " ").title()}')
        
        fig.update_layout(
            xaxis_title=group_by.replace("_", " ").title(),
            yaxis_title=metric.upper(),
            xaxis_tickangle=-45
        )
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{metric}_comparison_{group_by}.html')
        
        fig.write_html(save_path)
        
        self.logger.info(f"Saved interactive comparison plot: {save_path}")
        return save_path
    
    def plot_scatter_analysis(self, x_metric: str, y_metric: str,
                            color_by: Optional[str] = None,
                            size_by: Optional[str] = None,
                            filter_by: Optional[Dict[str, Any]] = None,
                            save_path: Optional[str] = None,
                            interactive: bool = False) -> str:
        """
        Create scatter plot for analyzing relationship between two metrics.
        
        Args:
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            color_by: Column to color points by
            size_by: Column to size points by
            filter_by: Dictionary of filters to apply
            save_path: Path to save the plot
            interactive: Whether to create interactive plot
            
        Returns:
            Path to saved plot
        """
        df = self.df.copy()
        
        # Apply filters
        if filter_by:
            for key, value in filter_by.items():
                if key in df.columns:
                    df = df[df[key] == value]
        
        if (df.empty or x_metric not in df.columns or y_metric not in df.columns):
            self.logger.warning(f"No data available for metrics: {x_metric}, {y_metric}")
            return ""
        
        if interactive:
            return self._plot_interactive_scatter(df, x_metric, y_metric, color_by, size_by, save_path)
        else:
            return self._plot_static_scatter(df, x_metric, y_metric, color_by, size_by, save_path)
    
    def _plot_static_scatter(self, df: pd.DataFrame, x_metric: str, y_metric: str,
                           color_by: Optional[str], size_by: Optional[str],
                           save_path: Optional[str]) -> str:
        """Create static scatter plot"""
        plt.figure(figsize=(12, 8))
        
        if color_by and color_by in df.columns:
            scatter = plt.scatter(df[x_metric], df[y_metric], c=df[color_by].astype('category').cat.codes,
                                cmap='tab10', alpha=0.7, s=60)
            plt.colorbar(scatter, label=color_by.replace("_", " ").title())
        else:
            plt.scatter(df[x_metric], df[y_metric], alpha=0.7, s=60)
        
        plt.xlabel(x_metric.replace("_", " ").title())
        plt.ylabel(y_metric.replace("_", " ").title())
        plt.title(f'{y_metric.upper()} vs {x_metric.upper()}')
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{y_metric}_vs_{x_metric}_scatter.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved scatter plot: {save_path}")
        return save_path
    
    def _plot_interactive_scatter(self, df: pd.DataFrame, x_metric: str, y_metric: str,
                                color_by: Optional[str], size_by: Optional[str],
                                save_path: Optional[str]) -> str:
        """Create interactive scatter plot"""
        fig = px.scatter(df, x=x_metric, y=y_metric,
                        color=color_by, size=size_by,
                        hover_data=['index_name', 'dataset_name', 'hardware_type'],
                        title=f'{y_metric.upper()} vs {x_metric.upper()}')
        
        fig.update_layout(
            xaxis_title=x_metric.replace("_", " ").title(),
            yaxis_title=y_metric.replace("_", " ").title()
        )
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{y_metric}_vs_{x_metric}_scatter.html')
        
        fig.write_html(save_path)
        
        self.logger.info(f"Saved interactive scatter plot: {save_path}")
        return save_path
    
    def plot_pareto_frontier(self, x_metric: str = 'search_time',
                           y_metric: str = 'recall',
                           color_by: str = 'index_name',
                           filter_by: Optional[Dict[str, Any]] = None,
                           save_path: Optional[str] = None,
                           interactive: bool = False) -> str:
        """
        Plot Pareto frontier for two competing metrics.
        
        Args:
            x_metric: Metric for x-axis (usually cost/time)
            y_metric: Metric for y-axis (usually quality)
            color_by: Column to color points by
            filter_by: Dictionary of filters to apply
            save_path: Path to save the plot
            interactive: Whether to create interactive plot
            
        Returns:
            Path to saved plot
        """
        from .analyzer import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(self.results)
        
        # Apply filters to results
        filtered_results = self.results
        if filter_by:
            filtered_results = self.results.filter_results(**filter_by)
        
        # Get Pareto optimal points
        pareto_df = analyzer.find_pareto_optimal(x_metric, y_metric)
        
        if pareto_df.empty:
            self.logger.warning("No Pareto optimal points found")
            return ""
        
        df = filtered_results.to_dataframe()
        
        if interactive:
            return self._plot_interactive_pareto(df, pareto_df, x_metric, y_metric, color_by, save_path)
        else:
            return self._plot_static_pareto(df, pareto_df, x_metric, y_metric, color_by, save_path)
    
    def _plot_static_pareto(self, df: pd.DataFrame, pareto_df: pd.DataFrame,
                          x_metric: str, y_metric: str, color_by: str,
                          save_path: Optional[str]) -> str:
        """Create static Pareto frontier plot"""
        plt.figure(figsize=(12, 8))
        
        # Plot all points
        if color_by in df.columns:
            for i, (name, group) in enumerate(df.groupby(color_by)):
                plt.scatter(group[x_metric], group[y_metric], 
                          label=name, alpha=0.6, s=60)
        else:
            plt.scatter(df[x_metric], df[y_metric], alpha=0.6, s=60, label='All points')
        
        # Plot Pareto frontier
        pareto_sorted = pareto_df.sort_values(x_metric)
        plt.plot(pareto_sorted[x_metric], pareto_sorted[y_metric], 
                'r-', linewidth=2, label='Pareto Frontier')
        plt.scatter(pareto_sorted[x_metric], pareto_sorted[y_metric], 
                   c='red', s=100, marker='*', label='Pareto Optimal')
        
        plt.xlabel(x_metric.replace("_", " ").title())
        plt.ylabel(y_metric.replace("_", " ").title())
        plt.title(f'Pareto Frontier: {y_metric.upper()} vs {x_metric.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'pareto_{y_metric}_vs_{x_metric}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved Pareto frontier plot: {save_path}")
        return save_path
    
    def _plot_interactive_pareto(self, df: pd.DataFrame, pareto_df: pd.DataFrame,
                               x_metric: str, y_metric: str, color_by: str,
                               save_path: Optional[str]) -> str:
        """Create interactive Pareto frontier plot"""
        fig = go.Figure()
        
        # Add all points
        if color_by in df.columns:
            for name, group in df.groupby(color_by):
                fig.add_trace(go.Scatter(
                    x=group[x_metric], y=group[y_metric],
                    mode='markers',
                    name=str(name),
                    opacity=0.6,
                    hovertemplate=f'{color_by}: %{{fullData.name}}<br>' +
                                 f'{x_metric}: %{{x}}<br>' +
                                 f'{y_metric}: %{{y}}<extra></extra>'
                ))
        
        # Add Pareto frontier
        pareto_sorted = pareto_df.sort_values(x_metric)
        fig.add_trace(go.Scatter(
            x=pareto_sorted[x_metric], y=pareto_sorted[y_metric],
            mode='lines+markers',
            name='Pareto Frontier',
            line=dict(color='red', width=3),
            marker=dict(color='red', size=10, symbol='star')
        ))
        
        fig.update_layout(
            title=f'Pareto Frontier: {y_metric.upper()} vs {x_metric.upper()}',
            xaxis_title=x_metric.replace("_", " ").title(),
            yaxis_title=y_metric.replace("_", " ").title()
        )
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'pareto_{y_metric}_vs_{x_metric}.html')
        
        fig.write_html(save_path)
        
        self.logger.info(f"Saved interactive Pareto frontier plot: {save_path}")
        return save_path
    
    def plot_hardware_comparison(self, metrics: List[str] = None,
                               save_path: Optional[str] = None,
                               interactive: bool = False) -> str:
        """
        Plot comparison between CPU and GPU performance.
        
        Args:
            metrics: List of metrics to compare
            save_path: Path to save the plot
            interactive: Whether to create interactive plot
            
        Returns:
            Path to saved plot
        """
        if 'hardware_type' not in self.df.columns:
            self.logger.warning("No hardware_type column found")
            return ""
        
        if metrics is None:
            metrics = ['qps', 'search_time', 'recall']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        if not available_metrics:
            self.logger.warning("No specified metrics found in data")
            return ""
        
        if interactive:
            return self._plot_interactive_hardware_comparison(available_metrics, save_path)
        else:
            return self._plot_static_hardware_comparison(available_metrics, save_path)
    
    def _plot_static_hardware_comparison(self, metrics: List[str],
                                       save_path: Optional[str]) -> str:
        """Create static hardware comparison plot"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            sns.boxplot(data=self.df, x='hardware_type', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} by Hardware Type')
            axes[i].set_ylabel(metric.upper())
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'hardware_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved hardware comparison plot: {save_path}")
        return save_path
    
    def _plot_interactive_hardware_comparison(self, metrics: List[str],
                                            save_path: Optional[str]) -> str:
        """Create interactive hardware comparison plot"""
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[m.upper() for m in metrics]
        )
        
        for i, metric in enumerate(metrics):
            for hardware_type in self.df['hardware_type'].unique():
                data = self.df[self.df['hardware_type'] == hardware_type][metric].dropna()
                
                fig.add_trace(
                    go.Box(y=data, name=f'{hardware_type.upper()}', 
                          showlegend=(i == 0)),
                    row=1, col=i+1
                )
        
        fig.update_layout(title='Hardware Performance Comparison')
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'hardware_comparison.html')
        
        fig.write_html(save_path)
        
        self.logger.info(f"Saved interactive hardware comparison plot: {save_path}")
        return save_path
    
    def plot_algorithm_ranking(self, metrics: List[str] = None,
                             weights: List[float] = None,
                             save_path: Optional[str] = None) -> str:
        """
        Plot algorithm ranking based on multiple metrics.
        
        Args:
            metrics: List of metrics for ranking
            weights: Weights for each metric
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        from .analyzer import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(self.results)
        ranking_df = analyzer.generate_ranking(metrics, weights)
        
        if ranking_df.empty:
            self.logger.warning("No ranking data available")
            return ""
        
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar plot
        y_pos = np.arange(len(ranking_df))
        plt.barh(y_pos, ranking_df['weighted_score'], alpha=0.7)
        plt.yticks(y_pos, ranking_df.index)
        plt.xlabel('Weighted Score')
        plt.title('Algorithm Ranking')
        plt.gca().invert_yaxis()
        
        # Add score labels
        for i, score in enumerate(ranking_df['weighted_score']):
            plt.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'algorithm_ranking.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved algorithm ranking plot: {save_path}")
        return save_path
    
    def create_dashboard(self, save_path: Optional[str] = None) -> str:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            save_path: Path to save the dashboard
            
        Returns:
            Path to saved dashboard
        """
        if self.df.empty:
            self.logger.warning("No data available for dashboard")
            return ""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['QPS Comparison', 'Recall vs Search Time', 
                          'Hardware Comparison', 'Memory Usage'],
            specs=[[{"type": "box"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        # QPS comparison
        if 'qps' in self.df.columns and 'index_name' in self.df.columns:
            for algorithm in self.df['index_name'].unique():
                data = self.df[self.df['index_name'] == algorithm]['qps'].dropna()
                fig.add_trace(
                    go.Box(y=data, name=algorithm, showlegend=False),
                    row=1, col=1
                )
        
        # Recall vs Search Time
        if 'recall' in self.df.columns and 'search_time' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['search_time'], y=self.df['recall'],
                    mode='markers',
                    text=self.df.get('index_name', ''),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Hardware comparison
        if 'hardware_type' in self.df.columns and 'qps' in self.df.columns:
            for hardware in self.df['hardware_type'].unique():
                data = self.df[self.df['hardware_type'] == hardware]['qps'].dropna()
                fig.add_trace(
                    go.Box(y=data, name=hardware, showlegend=False),
                    row=2, col=1
                )
        
        # Memory usage
        if 'memory_usage_rss_mb' in self.df.columns and 'index_name' in self.df.columns:
            memory_by_algorithm = self.df.groupby('index_name')['memory_usage_rss_mb'].mean()
            fig.add_trace(
                go.Bar(
                    x=memory_by_algorithm.index,
                    y=memory_by_algorithm.values,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='FAISS Benchmark Dashboard',
            height=800
        )
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'benchmark_dashboard.html')
        
        fig.write_html(save_path)
        
        self.logger.info(f"Saved benchmark dashboard: {save_path}")
        return save_path