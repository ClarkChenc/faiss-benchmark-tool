import time
import numpy as np
import faiss

def build_index(index, xb):
    """Trains the index and adds vectors."""
    t0 = time.time()
    index.train(xb)
    train_time = time.time() - t0

    t0 = time.time()
    index.add(xb)
    add_time = time.time() - t0

    return {"train_time": train_time, "add_time": add_time}

def search_index(index, xq, gt, topk=10, params=None):
    """Searches the index and returns performance metrics."""
    if params and "efSearch" in params:
        try:
            index.hnsw.efSearch = params["efSearch"]
        except AttributeError:
            pass  # Not an HNSW index

    t0 = time.time()
    D, I = index.search(xq, topk)
    search_time = time.time() - t0

    n_ok = 0
    for i in range(xq.shape[0]):
        n_ok += len(np.intersect1d(I[i, :topk], gt[i, :topk]))
    
    recall = n_ok / (len(xq) * topk)
    qps = len(xq) / search_time

    return {"search_time": search_time, "qps": qps, "recall": recall}