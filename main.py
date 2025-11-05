import argparse
import yaml
import os
import json


def split_params(params, index_type):
    """Split index params into build-time and search-time params.

    - Build params affect index structure and should be part of cache key.
    - Search params affect query behavior and should NOT change cache key.
    """
    params = params or {}
    build_params = {}
    search_params = {}

    # Common search-time params (latency_batch_size moved to global config)
    search_keys = {"efSearch", "nprobe"}

    # Build-time params by index type
    if "HNSW" in index_type:
        # HNSW build param controls graph construction breadth
        if "efConstruction" in params:
            build_params["efConstruction"] = params["efConstruction"]
    if "IVF" in index_type:
        # nlist is a build-time param if provided explicitly (index_type may already encode it)
        if "nlist" in params:
            build_params["nlist"] = params["nlist"]
    if "CAGRA" in index_type.upper():
        # CAGRA graph construction params (GPU-only index build)
        if "graph_degree" in params:
            build_params["graph_degree"] = params["graph_degree"]
        if "intermediate_graph_degree" in params:
            build_params["intermediate_graph_degree"] = params["intermediate_graph_degree"]
        if "metric" in params:
            build_params["metric"] = params["metric"]

    # Any other keys not explicitly recognized default to search params,
    # except for 'use_gpu' which is handled separately
    for k, v in params.items():
        if k == "use_gpu":
            continue
        elif k in build_params:
            continue
        elif k in search_keys:
            search_params[k] = v
        else:
            # Default unknown params to search-time to avoid cache churn
            search_params[k] = v

    return build_params, search_params

def main():
    parser = argparse.ArgumentParser(description="Faiss Benchmark Tool")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Set thread env limits BEFORE importing Faiss/Numpy-backed libs
    # Read minimal config for num_threads
    num_threads_env = 1
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                cfg_preview = yaml.safe_load(f) or {}
                num_threads_env = int(cfg_preview.get("num_threads", num_threads_env))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads_env))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads_env))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads_env))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads_env))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(num_threads_env))
    
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OPENBLAS_DYNAMIC", "0")
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

    # Import heavy libs after env setup so they honor thread limits
    import faiss
    from faiss_benchmark.datasets import load_dataset, get_dataset_info
    from faiss_benchmark.indexes import create_index
    from faiss_benchmark.benchmark import build_index, build_index_batch, search_index
    from faiss_benchmark.results import print_results
    from faiss_benchmark.utils import load_config

    # Load full configuration
    config = load_config(args.config)

    # Set number of threads for Faiss
    num_threads = int(config.get("num_threads", num_threads_env))
    try:
        faiss.omp_set_num_threads(num_threads)
    except Exception as _thread_err:
        print(f"Warning: failed to set Faiss threads: {_thread_err}")
    print(f"Using {faiss.omp_get_max_threads()} threads for Faiss")

    dataset_name = config["dataset"]
    index_types = config["index_types"]
    topk = config.get("topk", 10)
    # Global latency micro-batch size for per-query latency measurement (default 32)
    global_latency_bs = int(config.get("latency_batch_size", 32))
    # Global warmup queries to stabilize QPS/latency before timing
    global_warmup = int(config.get("warmup_queries", 0))
    
    # 获取批处理配置
    batch_config = config.get("batch_processing", {"enabled": False})
    use_batch_processing = batch_config.get("enabled", False)

    print(f"Loading dataset: {dataset_name}")
    try:
        if use_batch_processing:
            # 批处理模式：只获取数据集信息，不加载全部数据
            print("使用批处理模式（内存优化）")
            dataset_info = get_dataset_info(dataset_name)
            print(f"Dataset info loaded successfully!")
            print(f"Base vectors: {dataset_info['base_count']:,} x {dataset_info['dimension']}")
            print(f"Query vectors: {dataset_info['query_count']} x {dataset_info['dimension']}")
            print(f"Ground truth: {dataset_info['groundtruth'].shape}")
            
            dimension = dataset_info['dimension']
            xq = dataset_info['query_vectors']
            gt = dataset_info['groundtruth']
            xb = None  # 不加载到内存
        else:
            # 传统模式：加载全部数据到内存
            print("使用传统模式（全量加载）")
            xb, xq, gt = load_dataset(dataset_name)
            print(f"Dataset loaded successfully!")
            print(f"Base vectors: {xb.shape}")
            print(f"Query vectors: {xq.shape}")
            print(f"Ground truth: {gt.shape}")
            
            dimension = xb.shape[1]
            dataset_info = None
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    cache_dir = "index_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\nRunning benchmarks with {len(index_types)} index types...")
    print("=" * 50)

    for index_config in index_types:
        index_type = index_config["index_type"]
        # New schema: explicit index/search params
        explicit_build = index_config.get("index_param") or {}
        explicit_search = index_config.get("search_param") or {}
        # Backward-compat: legacy 'params'
        legacy_params = index_config.get("params", {})

        # Determine use_gpu (prefer top-level, then index_param, then legacy params)
        use_gpu = bool(index_config.get("use_gpu", explicit_build.get("use_gpu", legacy_params.get("use_gpu", False))))

        # If explicit provided, use them; otherwise split legacy
        if explicit_build or explicit_search:
            build_params = {k: v for k, v in explicit_build.items() if k != "use_gpu"}
            search_params = explicit_search
        else:
            build_params, search_params = split_params(legacy_params, index_type)

        build_param_str = "_".join([f"{k}{v}" for k, v in build_params.items()])
        gpu_str = "_gpu" if use_gpu else ""
        cache_filename = f"{dataset_name}_{index_type}_{build_param_str}{gpu_str}.index"
        cache_path = os.path.join(cache_dir, cache_filename)
        meta_path = os.path.join(cache_dir, cache_filename.replace('.index', '_meta.json'))

        # Show effective params after split
        print(f"\nTesting index: {index_type} | build_param={build_params} | search_param={search_params} | use_gpu={use_gpu}")

        try:
            if not use_gpu and os.path.exists(cache_path) and os.path.exists(meta_path):
                print(f"Loading index from cache: {cache_path}")
                index = faiss.read_index(cache_path)
                with open(meta_path, 'r') as f:
                    build_results = json.load(f)
                print(f"Using cached build times: train={build_results['train_time']:.3f}s, add={build_results['add_time']:.3f}s")
            else:
                if use_gpu:
                    print("GPU mode is enabled. Index will be rebuilt and not cached.")
                print("Building new index...")
                # Create index using ONLY build-time params to avoid search-param cache churn
                index = create_index(index_type, dimension, use_gpu=use_gpu, params=build_params)
                
                # 根据是否使用批处理选择构建方法
                if use_batch_processing:
                    build_results = build_index_batch(index, dataset_info, batch_config)
                else:
                    build_results = build_index(index, xb)
                
                if not use_gpu:
                    print(f"Saving index to cache: {cache_path}")
                    faiss.write_index(index, cache_path)
                    with open(meta_path, 'w') as f:
                        json.dump(build_results, f)

            # Apply search-time params during search (e.g., nprobe, efSearch). Latency micro-batch size from global config.
            # Warm-up queries help stabilize QPS and latency before timing.
            search_results = search_index(
                index,
                xq,
                gt,
                topk=topk,
                params=search_params,
                latency_batch_size=global_latency_bs,
                warmup_queries=global_warmup,
            )
            results = {**build_results, **search_results}
            print_results(index_type, results, topk=topk)

        except Exception as e:
            print(f"Error benchmarking {index_type}: {e}")
            continue

if __name__ == "__main__":
    main()
