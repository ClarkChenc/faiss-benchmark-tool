#!/usr/bin/env python3
"""
Advanced benchmark example for FAISS framework

This script demonstrates advanced features including:
- Multiple datasets
- Parameter sweeping
- Pareto frontier analysis
- Comprehensive reporting
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any

# Add the parent directory to the path to import faiss_benchmark
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faiss_benchmark import BenchmarkRunner, DatasetManager, Config
from faiss_benchmark.utils.io_utils import write_fvecs, write_ivecs
from faiss_benchmark.visualization import BenchmarkPlotter, ResultsAnalyzer


def create_multiple_datasets(data_dir: str = "datasets") -> List[str]:
    """
    Create multiple datasets with different characteristics.
    
    Args:
        data_dir: Directory to save datasets
        
    Returns:
        List of dataset names
    """
    datasets = [
        {"name": "small", "n_base": 5000, "n_query": 50, "dimension": 64},
        {"name": "medium", "n_base": 20000, "n_query": 100, "dimension": 128},
        {"name": "large", "n_base": 50000, "n_query": 200, "dimension": 256}
    ]
    
    os.makedirs(data_dir, exist_ok=True)
    created_datasets = []
    
    for dataset_config in datasets:
        name = dataset_config["name"]
        print(f"Creating dataset: {name}")
        
        # Check if dataset already exists
        base_file = os.path.join(data_dir, f"{name}_base.fvecs")
        if os.path.exists(base_file):
            print(f"  Dataset {name} already exists, skipping...")
            created_datasets.append(name)
            continue
        
        # Generate data
        np.random.seed(hash(name) % 2**32)  # Different seed for each dataset
        
        n_base = dataset_config["n_base"]
        n_query = dataset_config["n_query"]
        dimension = dataset_config["dimension"]
        
        # Create clustered data for more realistic scenarios
        n_clusters = max(10, n_base // 1000)
        cluster_centers = np.random.randn(n_clusters, dimension).astype(np.float32)
        
        # Generate base vectors around clusters
        base_vectors = []
        for i in range(n_base):
            cluster_id = np.random.randint(n_clusters)
            noise = np.random.randn(dimension).astype(np.float32) * 0.5
            vector = cluster_centers[cluster_id] + noise
            base_vectors.append(vector)
        base_vectors = np.array(base_vectors)
        
        # Generate query vectors (some near clusters, some random)
        query_vectors = []
        for i in range(n_query):
            if np.random.random() < 0.7:  # 70% near clusters
                cluster_id = np.random.randint(n_clusters)
                noise = np.random.randn(dimension).astype(np.float32) * 0.3
                vector = cluster_centers[cluster_id] + noise
            else:  # 30% random
                vector = np.random.randn(dimension).astype(np.float32)
            query_vectors.append(vector)
        query_vectors = np.array(query_vectors)
        
        # Generate ground truth
        print(f"  Generating ground truth for {name}...")
        distances = np.linalg.norm(
            base_vectors[np.newaxis, :, :] - query_vectors[:, np.newaxis, :], 
            axis=2
        )
        ground_truth = np.argsort(distances, axis=1)[:, :100].astype(np.int32)
        
        # Save files
        query_file = os.path.join(data_dir, f"{name}_query.fvecs")
        gt_file = os.path.join(data_dir, f"{name}_groundtruth.ivecs")
        
        write_fvecs(base_file, base_vectors)
        write_fvecs(query_file, query_vectors)
        write_ivecs(gt_file, ground_truth)
        
        print(f"  Saved {name}: {n_base} base, {n_query} query, dim={dimension}")
        created_datasets.append(name)
    
    return created_datasets


def run_parameter_sweep(runner: BenchmarkRunner, dataset_name: str) -> List[Dict[str, Any]]:
    """
    Run parameter sweep for IVF indexes.
    
    Args:
        runner: BenchmarkRunner instance
        dataset_name: Name of dataset to test
        
    Returns:
        List of results
    """
    print(f"Running parameter sweep on {dataset_name}...")
    
    # Define parameter ranges
    nlist_values = [50, 100, 200, 500]
    nprobe_values = [1, 5, 10, 20]
    
    results = []
    
    for nlist in nlist_values:
        for nprobe in nprobe_values:
            if nprobe > nlist:  # Skip invalid combinations
                continue
            
            print(f"  Testing nlist={nlist}, nprobe={nprobe}")
            
            index_config = {
                "type": "IVFFlat",
                "params": {
                    "nlist": nlist,
                    "nprobe": nprobe
                }
            }
            
            try:
                result = runner.run_single_benchmark(
                    dataset_name=dataset_name,
                    index_config=index_config,
                    hardware_type="cpu",
                    k=10
                )
                
                # Add parameter info to result
                result["nlist"] = nlist
                result["nprobe"] = nprobe
                results.append(result)
                
                print(f"    QPS: {result.get('qps', 0):.2f}, "
                      f"Recall: {result.get('recall', 0):.3f}")
                
            except Exception as e:
                print(f"    Failed: {str(e)}")
    
    return results


def analyze_pareto_frontier(results_manager, dataset_name: str = None):
    """
    Analyze and visualize Pareto frontier.
    
    Args:
        results_manager: BenchmarkResults instance
        dataset_name: Optional dataset filter
    """
    print("\n=== Pareto Frontier Analysis ===")
    
    analyzer = ResultsAnalyzer(results_manager)
    
    # Filter by dataset if specified
    filtered_results = results_manager
    if dataset_name:
        filtered_results = results_manager.filter_results(dataset_name=dataset_name)
        analyzer = ResultsAnalyzer(filtered_results)
    
    # Find Pareto optimal solutions
    pareto_df = analyzer.find_pareto_optimal(
        x_metric='search_time',
        y_metric='recall',
        minimize_x=True,
        maximize_y=True
    )
    
    if not pareto_df.empty:
        print(f"Found {len(pareto_df)} Pareto optimal solutions:")
        print(pareto_df[['index_name', 'search_time', 'recall', 'qps']].to_string(index=False))
        
        # Create Pareto frontier plot
        plotter = BenchmarkPlotter(filtered_results, output_dir="plots")
        pareto_plot = plotter.plot_pareto_frontier(
            x_metric='search_time',
            y_metric='recall',
            color_by='index_name',
            save_path=f"plots/pareto_frontier_{dataset_name or 'all'}.png"
        )
        print(f"Pareto frontier plot saved: {pareto_plot}")
        
        # Interactive version
        interactive_plot = plotter.plot_pareto_frontier(
            x_metric='search_time',
            y_metric='recall',
            color_by='index_name',
            save_path=f"plots/pareto_frontier_{dataset_name or 'all'}.html",
            interactive=True
        )
        print(f"Interactive Pareto plot saved: {interactive_plot}")
    else:
        print("No Pareto optimal solutions found")


def generate_comprehensive_report(results_manager):
    """
    Generate comprehensive analysis report.
    
    Args:
        results_manager: BenchmarkResults instance
    """
    print("\n=== Comprehensive Analysis Report ===")
    
    analyzer = ResultsAnalyzer(results_manager)
    
    # Summary statistics
    summary = analyzer.get_summary_report()
    print(f"Total results: {summary['overview']['total_results']}")
    print(f"Datasets tested: {summary['overview']['datasets']}")
    print(f"Algorithms tested: {summary['overview']['algorithms']}")
    print(f"Hardware types: {summary['overview']['hardware_types']}")
    
    # Best performers
    if 'best_qps' in summary:
        best_qps = summary['best_qps']
        print(f"\nBest QPS: {best_qps['algorithm']} on {best_qps['dataset']} "
              f"({best_qps['value']:.2f} QPS)")
    
    if 'best_recall' in summary:
        best_recall = summary['best_recall']
        print(f"Best Recall: {best_recall['algorithm']} on {best_recall['dataset']} "
              f"({best_recall['value']:.3f})")
    
    # Algorithm comparison
    print("\n=== Algorithm Performance Analysis ===")
    comparison = analyzer.compare_algorithms(metrics=['qps', 'recall', 'search_time'])
    if not comparison.empty:
        print(comparison.round(3))
    
    # Hardware efficiency analysis
    hw_analysis = analyzer.analyze_hardware_efficiency()
    if hw_analysis.get('comparisons'):
        print("\n=== Hardware Efficiency Analysis ===")
        for metric, stats in hw_analysis['comparisons'].items():
            speedup = stats.get('speedup')
            if speedup:
                print(f"{metric}: GPU is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than CPU")
    
    # Generate ranking
    ranking = analyzer.generate_ranking(
        metrics=['qps', 'recall'],
        weights=[0.6, 0.4],  # 60% weight on QPS, 40% on recall
        higher_is_better=[True, True]
    )
    
    if not ranking.empty:
        print("\n=== Overall Algorithm Ranking ===")
        print("(60% QPS + 40% Recall)")
        for i, (algorithm, row) in enumerate(ranking.iterrows()):
            print(f"{i+1}. {algorithm}: {row['weighted_score']:.3f}")


def run_advanced_benchmark():
    """Run advanced benchmark with multiple datasets and analysis"""
    
    print("=== FAISS Benchmark Framework - Advanced Example ===\n")
    
    # Create multiple datasets
    print("Setting up datasets...")
    dataset_names = create_multiple_datasets("datasets")
    print(f"Available datasets: {dataset_names}\n")
    
    # Initialize benchmark runner
    print("Initializing benchmark runner...")
    runner = BenchmarkRunner("config.yaml")
    
    # Add datasets to runner
    for dataset_name in dataset_names:
        runner.dataset_manager.add_dataset(
            name=dataset_name,
            base_file=f"datasets/{dataset_name}_base.fvecs",
            query_file=f"datasets/{dataset_name}_query.fvecs",
            ground_truth_file=f"datasets/{dataset_name}_groundtruth.ivecs"
        )
    
    # Define comprehensive index configurations
    index_configs = [
        {"type": "Flat", "params": {}},
        {"type": "IVFFlat", "params": {"nlist": 100}},
        {"type": "IVFFlat", "params": {"nlist": 200}},
        {"type": "IVFPQ", "params": {"nlist": 100, "m": 8, "nbits": 8}},
        {"type": "IVFPQ", "params": {"nlist": 200, "m": 16, "nbits": 8}},
        {"type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
        {"type": "LSH", "params": {"nbits": 1024}}
    ]
    
    print(f"Testing {len(index_configs)} index configurations...")
    print("Index types:", [f"{config['type']}({config['params']})" for config in index_configs])
    print()
    
    # Run benchmarks on all datasets
    all_results = []
    for dataset_name in dataset_names:
        print(f"=== Benchmarking dataset: {dataset_name} ===")
        
        dataset_results = runner.run_dataset_benchmark(
            dataset_name=dataset_name,
            index_configs=index_configs,
            hardware_types=["cpu"],  # Focus on CPU for this example
            k_values=[10]
        )
        
        all_results.extend(dataset_results)
        print(f"Completed {len(dataset_results)} tests for {dataset_name}\n")
    
    # Run parameter sweep on medium dataset
    if "medium" in dataset_names:
        print("=== Parameter Sweep Analysis ===")
        sweep_results = run_parameter_sweep(runner, "medium")
        all_results.extend(sweep_results)
        print(f"Parameter sweep completed: {len(sweep_results)} configurations tested\n")
    
    # Save all results
    print("Saving results...")
    results_manager = runner.get_results_manager()
    json_file = results_manager.save_results("advanced_benchmark_results.json")
    csv_file = results_manager.save_csv("advanced_benchmark_results.csv")
    print(f"Results saved to {json_file} and {csv_file}\n")
    
    # Generate comprehensive analysis
    generate_comprehensive_report(results_manager)
    
    # Pareto frontier analysis for each dataset
    for dataset_name in dataset_names:
        analyze_pareto_frontier(results_manager, dataset_name)
    
    # Generate visualizations
    print("\n=== Generating Advanced Visualizations ===")
    try:
        plotter = BenchmarkPlotter(results_manager, output_dir="plots")
        
        # Performance comparison by dataset
        for dataset_name in dataset_names:
            plot_path = plotter.plot_performance_comparison(
                metric='qps',
                group_by='index_name',
                filter_by={'dataset_name': dataset_name},
                save_path=f"plots/qps_comparison_{dataset_name}.png"
            )
            print(f"QPS comparison for {dataset_name}: {plot_path}")
        
        # Scalability analysis
        scalability_plot = plotter.plot_scatter_analysis(
            x_metric='dataset_size',
            y_metric='qps',
            color_by='index_name',
            save_path="plots/scalability_analysis.png"
        )
        print(f"Scalability analysis: {scalability_plot}")
        
        # Algorithm ranking
        ranking_plot = plotter.plot_algorithm_ranking(
            metrics=['qps', 'recall'],
            weights=[0.6, 0.4],
            save_path="plots/algorithm_ranking.png"
        )
        print(f"Algorithm ranking: {ranking_plot}")
        
        # Comprehensive dashboard
        dashboard = plotter.create_dashboard("plots/advanced_dashboard.html")
        print(f"Advanced dashboard: {dashboard}")
        
        # Interactive comparison plots
        for metric in ['qps', 'recall', 'search_time']:
            interactive_plot = plotter.plot_performance_comparison(
                metric=metric,
                group_by='index_name',
                save_path=f"plots/{metric}_interactive_comparison.html",
                interactive=True
            )
            print(f"Interactive {metric} comparison: {interactive_plot}")
        
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")
        print("Install matplotlib, seaborn, and plotly for visualizations")
    
    print("\n=== Advanced Benchmark Complete ===")
    print(f"Total results: {len(all_results)}")
    print(f"Datasets tested: {len(dataset_names)}")
    print(f"Index configurations: {len(index_configs)}")
    print("\nCheck the 'results' and 'plots' directories for detailed output.")
    print("Open 'plots/advanced_dashboard.html' for interactive exploration.")


if __name__ == "__main__":
    run_advanced_benchmark()