def print_results(index_type, results, topk=10):
    """Prints the benchmark results in a formatted way."""
    print(f"\n--- Results for {index_type} ---")
    print(f"Training time: {results['train_time']:.4f} seconds")
    print(f"Adding time: {results['add_time']:.4f} seconds")
    print(f"Search time: {results['search_time']:.4f} seconds")
    print(f"QPS: {results['qps']:.2f}")
    # Latency metrics (optional keys)
    if 'latency_avg_ms' in results:
        print(f"Latency avg (ms): {results['latency_avg_ms']:.3f}")
    if 'latency_p99_ms' in results:
        print(f"Latency p99 (ms): {results['latency_p99_ms']:.3f}")
    # GPU memory metrics (peak only)
    if 'gpu_mem_peak_used_bytes' in results:
        to_gb = lambda b: b / (1024 ** 3) if b is not None else None
        used_gb = to_gb(results.get('gpu_mem_peak_used_bytes'))
        total_gb = to_gb(results.get('gpu_mem_total_bytes'))
        if used_gb is not None:
            if total_gb is not None:
                print(f"GPU memory peak: {used_gb:.2f}GB / {total_gb:.2f}GB")
            else:
                print(f"GPU memory peak: {used_gb:.2f}GB")
    recall_key = f"Recall@{topk}"
    print(f"{recall_key}: {results['recall']:.4f}")
    if 'hit_rate' in results:
        hit_info = results['hit_rate']
        if len(hit_info) == 3:
             hit, total, indegree_hit = hit_info
             if total > 0:
                print(f"Entrypoint-majority hit rate: {hit/total:.3f} ({hit}/{total})")
                print(f"Indegree node hit rate: {indegree_hit/total:.3f} ({indegree_hit}/{total})")
        elif len(hit_info) == 2:
            hit, total = hit_info
            if total > 0:
                print(f"Entrypoint-majority hit rate: {hit/total:.3f} ({hit}/{total})")
    print("-" * 40)