import time
import numpy as np
from .datasets import load_base_vectors_batch
from .gpu_mem import get_gpu_memory

def build_index(index, xb):
    """Trains the index and adds vectors."""
    t0 = time.time()
    # Allow adapters to pre-allocate capacity when possible
    try:
        if hasattr(index, "init_capacity"):
            index.init_capacity(int(xb.shape[0]))
    except Exception:
        pass

    train_size = min(int(xb.shape[0] * 0.1), 1000000)
    train_data = xb[:train_size]

    index.train(train_data)
    train_time = time.time() - t0

    print(f"begin to build_index")
    t0 = time.time()
    index.add(xb)
    # 某些适配器（如 ScaNN）需要在构建后调用 finalize_build 完成索引创建
    if hasattr(index, "finalize_build"):
        try:
            fb = index.finalize_build()
        except Exception as e:
            raise RuntimeError(f"finalize_build occurs error: {e}")
            
    add_time = time.time() - t0
    print(f"build_index done, time used: {add_time:.4f}s")
    mem_info = get_gpu_memory()
    peak_used = mem_info["used_bytes"] if mem_info else None
    total_bytes = mem_info["total_bytes"] if mem_info else None


    return {
        "train_time": train_time,
        "add_time": add_time,
        "gpu_mem_peak_used_bytes": peak_used,
        "gpu_mem_total_bytes": total_bytes,
    }

# 这个函数不兼容 gpu 下的索引，gpu 索引需要一次性全部 add
def build_index_batch(index, dataset_info, batch_config):
    """使用分批处理构建索引"""
    base_path = dataset_info['base_path']
    base_count = dataset_info['base_count']
    dimension = dataset_info['dimension']
    
    batch_size = batch_config['batch_size']
    train_batch_size = batch_config['train_batch_size']
    
    print(f"使用批处理模式构建索引:")
    print(f"  总向量数: {base_count:,}")
    print(f"  训练批大小: {train_batch_size:,}")
    print(f"  添加批大小: {batch_size:,}")
    
    # 第一步：训练索引（使用较大的批次）
    print("正在训练索引...")
    t0 = time.time()
    # Pre-allocate capacity for adapters that need it (e.g., hnswlib)
    try:
        if hasattr(index, "init_capacity"):
            index.init_capacity(int(base_count))
    except Exception:
        pass
    
    # 加载训练数据（使用前 train_batch_size 个向量）
    train_vectors = load_base_vectors_batch(base_path, 0, train_batch_size, dimension)
    if train_vectors is not None:
        index.train(train_vectors)
        del train_vectors  # 立即释放内存
    else:
        # 如果训练批次太大，回退到较小的批次
        print(f"训练批次太大，使用 {batch_size} 个向量进行训练")
        train_vectors = load_base_vectors_batch(base_path, 0, batch_size, dimension)
        if train_vectors is not None:
            index.train(train_vectors)
            del train_vectors
    
    train_time = time.time() - t0
    print(f"训练完成，耗时: {train_time:.3f}s")
    
    # 第二步：分批添加向量
    print("正在分批添加向量...")
    t0 = time.time()
    
    processed = 0
    batch_num = 0
    
    while processed < base_count:
        current_batch_size = min(batch_size, base_count - processed)
        
        # 加载当前批次
        batch_vectors = load_base_vectors_batch(base_path, processed, current_batch_size, dimension)
        
        if batch_vectors is None:
            break
            
        # 添加到索引
        index.add(batch_vectors)
        
        processed += batch_vectors.shape[0]
        batch_num += 1
        
        # 释放内存
        del batch_vectors
        
        # 进度报告
        if batch_num % 10 == 0 or processed >= base_count:
            progress = (processed / base_count) * 100
            print(f"  进度: {processed:,}/{base_count:,} ({progress:.1f}%)")
    
    add_time = time.time() - t0
    print(f"添加完成，耗时: {add_time:.3f}s")
    
    mem_info = get_gpu_memory()
    peak_used = mem_info["used_bytes"] if mem_info else None
    total_bytes = mem_info["total_bytes"] if mem_info else None
    # 某些适配器（如 ScaNN）需要在构建后调用 finalize_build 完成索引创建
    if hasattr(index, "finalize_build"):
        try:
            fb = index.finalize_build()
            add_time = (add_time or 0.0) + float(fb.get("add_time", 0.0))
            peak_used = peak_used if peak_used is not None else fb.get("gpu_mem_peak_used_bytes")
            total_bytes = total_bytes if total_bytes is not None else fb.get("gpu_mem_total_bytes")
        except Exception:
            pass
    return {
        "train_time": train_time,
        "add_time": add_time,
        "gpu_mem_peak_used_bytes": peak_used,
        "gpu_mem_total_bytes": total_bytes,
    }

def search_index(index, xq, gt, topk=10, params=None, latency_batch_size=None, warmup_queries: int | None = None, initial_gpu_peak_bytes=None, gpu_total_bytes=None):
    """Searches the index and returns performance metrics.

    In addition to total search time, computes per-query latency metrics (avg, p99).
    Latency is measured using micro-batches to avoid excessive overhead on large query sets.
    The micro-batch size is controlled by global `latency_batch_size` (config-level),
    and falls back to a legacy per-index param if provided.
    """
    if params:
        # Apply search-time params
        if "efSearch" in params:
            try:
                index.hnsw.efSearch = int(params["efSearch"])  # HNSW (CPU)
            except AttributeError:
                pass  # Not an HNSW index

        # IVF nprobe setting: prefer direct attribute for GPU IVF, fallback to ParameterSpace
        if "nprobe" in params and params["nprobe"] is not None:
            nprobe_val = int(params["nprobe"])  # normalize

            # Prefer setattr for GPU indices (and many CPU variants)
            try:
                if hasattr(index, "nprobe"):
                    print(f"set nprob: {nprobe_val}")
                    setattr(index, "nprobe", nprobe_val)
            except Exception:
                print(f"Failed to set nprobe={nprobe_val} via setattr")            # Fallback to ParameterSpace when attribute not available

        # IndexRefine re-ranking width (k_factor)
        if "k_factor" in params:
            try:
                k_factor = int(params["k_factor"])
                setattr(index, "k_factor",k_factor )
            except Exception:
                pass

    # Micro-batch size for latency measurement (default: 32)
    mb_size = 32 if latency_batch_size is None else int(latency_batch_size)
    # Backward-compat: if not provided globally, allow legacy param lookup
    if latency_batch_size is None and params and isinstance(params, dict):
        mb_size = int(params.get("latency_batch_size", mb_size))

    # Warm-up phase: run a number of queries to stabilize QPS/latency
    print(f"query warm up")
    warmup = int(warmup_queries or 0)
    if warmup > 0:
        w_processed = 0
        w_total = min(warmup, xq.shape[0])
        while w_processed < w_total:
            w_bs = min(mb_size, w_total - w_processed)
            batch_queries = xq[w_processed:w_processed + w_bs]
            if hasattr(index, "search_with_params"):
                _D, _I = index.search_with_params(batch_queries, topk, params)
            else:
                _D, _I = index.search(batch_queries, topk)
            w_processed += w_bs

    n_queries = xq.shape[0]

    total_search_time = 0.0
    mem_info_before = get_gpu_memory()
    before_used = mem_info_before["used_bytes"] if mem_info_before else None
    processed = 0

    # Perform search in micro-batches, recording per-query latency
    repeat = 1
    I_all = np.empty((n_queries*repeat, topk), dtype=gt.dtype)
    latencies = []
    print(f"begin to search, mb_size: {mb_size}, repeat {repeat}")
    offset = 0
    while processed < repeat*n_queries:
        bs = min(mb_size, repeat*n_queries - processed)
        t0 = time.time()
        # Prefer adapter method that can accept search params when available
        if offset + bs <= n_queries:
            batch_queries = xq[offset:offset + bs]
            offset = offset + bs
        else:
            half0 = xq[offset:n_queries]
            half1 = xq[0:bs - n_queries + offset]
            offset = bs - n_queries + offset
            batch_queries = np.concatenate((half0, half1))
            
        if hasattr(index, "search_with_params"):
            D, I = index.search_with_params(batch_queries, topk, params)
        else:
            D, I = index.search(batch_queries, topk)
        elapsed = time.time() - t0
        total_search_time += elapsed

        # Store results
        I_all[processed:processed + bs, :] = I

        # Approximate per-query latency by dividing batch time evenly
        per_query_latency = elapsed / bs
        latencies.extend([per_query_latency] * bs)
        
        processed += bs

    # Compute recall
    n_ok = 0
    for i in range(n_queries*repeat):
        n_ok += len(np.intersect1d(I_all[i, :topk], gt[i%n_queries, :topk]))
    recall = n_ok / (n_queries * repeat * topk)

    if hasattr(index, "get_stat"):
        index.get_stat(processed)

    # Throughput and latency metrics
    qps = n_queries * repeat / total_search_time if total_search_time > 0 else 0.0
    latencies_ms = np.array(latencies, dtype=np.float64) * 1000.0
    latency_avg_ms = float(latencies_ms.mean()) if latencies_ms.size > 0 else 0.0
    latency_p99_ms = float(np.percentile(latencies_ms, 99)) if latencies_ms.size > 0 else 0.0
    mem_info_after = get_gpu_memory()
    after_used = mem_info_after["used_bytes"] if mem_info_after else None
    # Compute peak used across provided snapshots
    candidates = [initial_gpu_peak_bytes, before_used, after_used]
    peak_used = None
    for c in candidates:
        if c is None:
            continue
        peak_used = c if peak_used is None else max(peak_used, c)
    # Prefer an available total bytes source
    total_bytes = gpu_total_bytes or (mem_info_before["total_bytes"] if mem_info_before else (mem_info_after["total_bytes"] if mem_info_after else None))

    return {
        "search_time": total_search_time,
        "qps": qps,
        "recall": recall,
        "latency_avg_ms": latency_avg_ms,
        "latency_p99_ms": latency_p99_ms,
        "gpu_mem_peak_used_bytes": peak_used,
        "gpu_mem_total_bytes": total_bytes,
    }
