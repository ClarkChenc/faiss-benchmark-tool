def print_results(index_type, results):
    """Prints the benchmark results in a formatted way."""
    print(f"\n--- Results for {index_type} ---")
    print(f"Training time: {results['train_time']:.4f} seconds")
    print(f"Adding time: {results['add_time']:.4f} seconds")
    print(f"Search time: {results['search_time']:.4f} seconds")
    print(f"QPS: {results['qps']:.2f}")
    print(f"Recall@10: {results['recall']:.4f}")
    print("-" * 40)