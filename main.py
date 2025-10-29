import argparse
import yaml
from faiss_benchmark.datasets import load_dataset
from faiss_benchmark.indexes import create_index
from faiss_benchmark.benchmark import run_benchmark
from faiss_benchmark.results import print_results

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Faiss Benchmark Tool")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    dataset_name = config["dataset"]
    index_types = config["index_types"]
    topk = config.get("topk", 10)

    print(f"Loading dataset: {dataset_name}")
    try:
        xb, xq, gt = load_dataset(dataset_name)
        print(f"Dataset loaded successfully!")
        print(f"Base vectors: {xb.shape}")
        print(f"Query vectors: {xq.shape}")
        print(f"Ground truth: {gt.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    dimension = xb.shape[1]

    print(f"\nRunning benchmarks with {len(index_types)} index types...")
    print("=" * 50)

    for index_type in index_types:
        print(f"\nTesting index: {index_type}")
        
        try:
            index = create_index(index_type, dimension, use_gpu=args.gpu)
            results = run_benchmark(index, xb, xq, gt, topk=topk)
            print_results(index_type, results, topk=topk)
        except Exception as e:
            print(f"Error with index {index_type}: {e}")
            continue

if __name__ == "__main__":
    main()
