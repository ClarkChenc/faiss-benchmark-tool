#!/usr/bin/env python3
"""
Basic benchmark example for FAISS framework

This script demonstrates how to run a simple benchmark comparing
different FAISS indexes on a dataset.
"""

import os
import sys
import numpy as np

# Add the parent directory to the path to import faiss_benchmark
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faiss_benchmark import BenchmarkRunner, DatasetManager, Config
from faiss_benchmark.utils.io_utils import write_fvecs, write_ivecs


def create_sample_dataset(name: str = "sample", 
                         n_base: int = 10000, 
                         n_query: int = 100, 
                         dimension: int = 128,
                         data_dir: str = "datasets") -> None:
    """
    Create a sample dataset for testing.
    
    Args:
        name: Dataset name
        n_base: Number of base vectors
        n_query: Number of query vectors
        dimension: Vector dimension
        data_dir: Directory to save dataset
    """
    print(f"Creating sample dataset: {name}")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate random vectors
    np.random.seed(42)  # For reproducibility
    base_vectors = np.random.randn(n_base, dimension).astype(np.float32)
    query_vectors = np.random.randn(n_query, dimension).astype(np.float32)
    
    # Generate ground truth (simple brute force)
    print("Generating ground truth...")
    distances = np.linalg.norm(base_vectors[np.newaxis, :, :] - query_vectors[:, np.newaxis, :], axis=2)
    ground_truth = np.argsort(distances, axis=1)[:, :100].astype(np.int32)  # Top 100 neighbors
    
    # Save to files
    base_file = os.path.join(data_dir, f"{name}_base.fvecs")
    query_file = os.path.join(data_dir, f"{name}_query.fvecs")
    gt_file = os.path.join(data_dir, f"{name}_groundtruth.ivecs")
    
    write_fvecs(base_file, base_vectors)
    write_fvecs(query_file, query_vectors)
    write_ivecs(gt_file, ground_truth)
    
    print(f"Dataset saved:")
    print(f"  Base vectors: {base_file} ({n_base} vectors)")
    print(f"  Query vectors: {query_file} ({n_query} vectors)")
    print(f"  Ground truth: {gt_file}")


def run_basic_benchmark():
    """Run a basic benchmark example"""
    
    print("=== FAISS Benchmark Framework - Basic Example ===\n")
    
    # Create sample dataset if it doesn't exist
    dataset_dir = "datasets"
    sample_base = os.path.join(dataset_dir, "sample_base.fvecs")
    
    if not os.path.exists(sample_base):
        print("Sample dataset not found. Creating one...")
        create_sample_dataset("sample", n_base=10000, n_query=100, dimension=128, data_dir=dataset_dir)
        print()
    
    # Initialize benchmark runner
    print("Initializing benchmark runner...")
    runner = BenchmarkRunner("config.yaml")
    
    # Add sample dataset to configuration
    print("Adding sample dataset...")
    runner.dataset_manager.add_dataset(
        name="sample",
        base_file=os.path.join(dataset_dir, "sample_base.fvecs"),
        query_file=os.path.join(dataset_dir, "sample_query.fvecs"),
        ground_truth_file=os.path.join(dataset_dir, "sample_groundtruth.ivecs")
    )
    
    # Define index configurations to test
    index_configs = [
        {
            "type": "Flat",
            "params": {}
        },
        {
            "type": "IVFFlat",
            "params": {
                "nlist": 100
            }
        },
        {
            "type": "IVFPQ",
            "params": {
                "nlist": 100,
                "m": 8,
                "nbits": 8
            }
        }
    ]
    
    print(f"Running benchmark with {len(index_configs)} index types...")
    print("Index types:", [config["type"] for config in index_configs])
    print()
    
    # Run benchmarks
    results = []
    for i, index_config in enumerate(index_configs):
        print(f"[{i+1}/{len(index_configs)}] Testing {index_config['type']}...")
        
        try:
            # Test on CPU
            result_cpu = runner.run_single_benchmark(
                dataset_name="sample",
                index_config=index_config,
                hardware_type="cpu",
                k=10
            )
            results.append(result_cpu)
            
            print(f"  CPU - QPS: {result_cpu.get('qps', 0):.2f}, "
                  f"Recall: {result_cpu.get('recall', 0):.3f}, "
                  f"Build time: {result_cpu.get('index_build_time', 0):.2f}s")
            
            # Test on GPU if available
            try:
                result_gpu = runner.run_single_benchmark(
                    dataset_name="sample",
                    index_config=index_config,
                    hardware_type="gpu",
                    k=10
                )
                results.append(result_gpu)
                
                print(f"  GPU - QPS: {result_gpu.get('qps', 0):.2f}, "
                      f"Recall: {result_gpu.get('recall', 0):.3f}, "
                      f"Build time: {result_gpu.get('index_build_time', 0):.2f}s")
                
            except Exception as e:
                print(f"  GPU test failed: {str(e)}")
        
        except Exception as e:
            print(f"  Failed: {str(e)}")
        
        print()
    
    # Save results
    print("Saving results...")
    results_manager = runner.get_results_manager()
    json_file = results_manager.save_results("basic_benchmark_results.json")
    csv_file = results_manager.save_csv("basic_benchmark_results.csv")
    
    print(f"Results saved to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")
    
    # Generate comparison table
    print("\n=== Performance Comparison ===")
    comparison_df = results_manager.get_comparison_table(
        metrics=['qps', 'recall', 'search_time', 'index_build_time'],
        group_by=['index_name', 'hardware_type']
    )
    
    if not comparison_df.empty:
        print(comparison_df.to_string(index=False))
    else:
        print("No comparison data available")
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    try:
        from faiss_benchmark.visualization import BenchmarkPlotter
        
        plotter = BenchmarkPlotter(results_manager, output_dir="plots")
        
        # QPS comparison
        qps_plot = plotter.plot_performance_comparison(
            metric='qps',
            group_by='index_name',
            save_path="plots/qps_comparison.png"
        )
        print(f"QPS comparison plot: {qps_plot}")
        
        # Recall vs Search Time scatter plot
        scatter_plot = plotter.plot_scatter_analysis(
            x_metric='search_time',
            y_metric='recall',
            color_by='index_name',
            save_path="plots/recall_vs_time.png"
        )
        print(f"Recall vs Time plot: {scatter_plot}")
        
        # Hardware comparison
        if any('gpu' in str(r.get('hardware_type', '')) for r in results):
            hw_plot = plotter.plot_hardware_comparison(
                metrics=['qps', 'search_time'],
                save_path="plots/hardware_comparison.png"
            )
            print(f"Hardware comparison plot: {hw_plot}")
        
        # Interactive dashboard
        dashboard = plotter.create_dashboard("plots/dashboard.html")
        print(f"Interactive dashboard: {dashboard}")
        
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")
        print("Install matplotlib, seaborn, and plotly for visualizations")
    
    print("\n=== Benchmark Complete ===")
    print(f"Total results: {len(results)}")
    print("Check the 'results' and 'plots' directories for detailed output.")


if __name__ == "__main__":
    run_basic_benchmark()