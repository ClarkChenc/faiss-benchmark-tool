def print_results(index_type: str, results: dict):
    """Prints the benchmark results in a formatted way."""
    print(f"  Index Type:    {index_type}")
    print(f"  Train Time:    {results['train_time']:.4f} s")
    print(f"  Add Time:      {results['add_time']:.4f} s")
    print(f"  Search Time:   {results['search_time']:.4f} s")
    print(f"  QPS:           {results['qps']:.2f}")