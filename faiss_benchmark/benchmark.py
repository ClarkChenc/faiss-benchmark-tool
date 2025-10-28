import time

def run_benchmark(index, xb, xq):
    """Runs the benchmark for a given index and dataset."""
    # Train the index
    t0 = time.time()
    index.train(xb)
    train_time = time.time() - t0

    # Add vectors to the index
    t0 = time.time()
    index.add(xb)
    add_time = time.time() - t0

    # Search the index
    t0 = time.time()
    index.search(xq, 10)
    search_time = time.time() - t0

    qps = len(xq) / search_time if search_time > 0 else 0

    return {
        "train_time": train_time,
        "add_time": add_time,
        "search_time": search_time,
        "qps": qps,
    }