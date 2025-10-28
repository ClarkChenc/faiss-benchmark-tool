import time
import numpy as np

def run_benchmark(index, xb, xq, gt, k=10):
    """Runs the benchmark and returns performance metrics."""
    # Training
    t0 = time.time()
    index.train(xb)
    train_time = time.time() - t0

    # Adding vectors
    t0 = time.time()
    index.add(xb)
    add_time = time.time() - t0

    # Searching
    t0 = time.time()
    D, I = index.search(xq, k)
    search_time = time.time() - t0

    # Calculate recall
    recall = (I[:, :k] == gt[:, :k]).sum() / (len(xq) * k)

    qps = len(xq) / search_time

    return {
        "train_time": train_time,
        "add_time": add_time,
        "search_time": search_time,
        "qps": qps,
        "recall": recall
    }