import time
import numpy as np
import faiss
from .datasets import load_base_vectors_batch

def build_index(index, xb):
    """Trains the index and adds vectors."""
    t0 = time.time()
    index.train(xb)
    train_time = time.time() - t0

    t0 = time.time()
    index.add(xb)
    add_time = time.time() - t0

    return {"train_time": train_time, "add_time": add_time}

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