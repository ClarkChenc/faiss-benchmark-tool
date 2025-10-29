import argparse
import yaml
import os
import json
import faiss
from faiss_benchmark.datasets import load_dataset
from faiss_benchmark.indexes import create_index
from faiss_benchmark.benchmark import build_index, search_index
from faiss_benchmark.results import print_results
from faiss_benchmark.utils import load_config

def main():
    parser = argparse.ArgumentParser(description="Faiss Benchmark Tool")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set number of threads for Faiss
    num_threads = config.get("num_threads", 1)
    faiss.omp_set_num_threads(num_threads)

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

    cache_dir = "index_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\nRunning benchmarks with {len(index_types)} index types...")
    print("=" * 50)

    for index_config in index_types:
        index_type = index_config["index_type"]
        params = index_config.get("params", {})
        param_str = "_".join([f"{k}{v}" for k, v in params.items()])
        cache_filename = f"{dataset_name}_{index_type}_{param_str}.index"
        cache_path = os.path.join(cache_dir, cache_filename)
        meta_path = os.path.join(cache_dir, cache_filename.replace('.index', '_meta.json'))

        print(f"\nTesting index: {index_type} with params: {params}")

        try:
            if os.path.exists(cache_path) and os.path.exists(meta_path):
                print(f"Loading index from cache: {cache_path}")
                index = faiss.read_index(cache_path)
                with open(meta_path, 'r') as f:
                    build_results = json.load(f)
                print(f"Using cached build times: train={build_results['train_time']:.3f}s, add={build_results['add_time']:.3f}s")
            else:
                print("Building new index...")
                index = create_index(index_type, dimension, use_gpu=args.gpu, params=params)
                build_results = build_index(index, xb)
                print(f"Saving index to cache: {cache_path}")
                faiss.write_index(index, cache_path)
                with open(meta_path, 'w') as f:
                    json.dump(build_results, f)

            search_results = search_index(index, xq, gt, topk=topk, params=params)
            results = {**build_results, **search_results}
            print_results(index_type, results, topk=topk)

        except Exception as e:
            print(f"Error benchmarking {index_type}: {e}")
            continue

if __name__ == "__main__":
    main()
