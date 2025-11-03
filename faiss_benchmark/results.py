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
    recall_key = f"Recall@{topk}"
    print(f"{recall_key}: {results['recall']:.4f}")
    print("-" * 40)