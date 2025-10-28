import argparse
from faiss_benchmark.datasets import create_random_dataset
from faiss_benchmark.indexes import create_index
from faiss_benchmark.benchmark import run_benchmark
from faiss_benchmark.results import print_results

# Configuration
DATASET_SIZE = 100000
DIMENSION = 128
INDEX_TYPES = [
    # Basic indexes
    "Flat",
    "IVFFlat",
    "LSH",
    # PQ-based indexes
    "IVFPQ",
    "PQ",
    # HNSW indexes
    "HNSWFlat",
    "HNSW,PQ32",
    # GPU-specific indexes (if GPU is available)
    "IVF1024,Flat",
    "IVF1024,PQ32"
]

def main(use_gpu: bool):
    """Main function to run the benchmark."""
    print(f"Running benchmark with {'GPU' if use_gpu else 'CPU'}")
    xb, xq = create_random_dataset(DATASET_SIZE, DIMENSION)

    for index_type in INDEX_TYPES:
        print(f"\n----- Benchmarking {index_type} -----")
        index = create_index(index_type, DIMENSION, use_gpu)
        if index is None:
            continue

        results = run_benchmark(index, xb, xq)
        print_results(index_type, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faiss Benchmark")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU support")
    args = parser.parse_args()
    main(args.gpu)