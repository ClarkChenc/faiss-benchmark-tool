#!/usr/bin/env python3
"""
数据集生成工具

将单个 .fvecs 文件分割成 query 和 base 两部分，并生成对应的 groundtruth 文件。
支持 GPU 加速和分批计算，适用于大规模数据集。
"""

import argparse
import os
import numpy as np
import faiss
from tqdm import tqdm
from faiss_benchmark.utils import fvecs_read, fvecs_write, ivecs_write


def split_dataset(input_file, num_queries, output_prefix):
    """
    将输入的 .fvecs 文件分割成 query 和 base 两部分
    
    Args:
        input_file: 输入的 .fvecs 文件路径
        num_queries: query 集合的向量数量
        output_prefix: 输出文件的前缀（不含扩展名）
    
    Returns:
        tuple: (query_vectors, base_vectors) 分别为 query 和 base 向量数组
    """
    print(f"正在读取数据文件: {input_file}")
    vectors = fvecs_read(input_file)
    
    if len(vectors) < num_queries:
        raise ValueError(f"数据集总数量 ({len(vectors)}) 小于请求的 query 数量 ({num_queries})")
    
    # 分割数据集
    query_vectors = vectors[:num_queries]
    base_vectors = vectors[num_queries:]
    
    # 保存分割后的文件
    query_file = f"{output_prefix}_query.fvecs"
    base_file = f"{output_prefix}_base.fvecs"
    
    print(f"保存 query 向量到: {query_file} ({len(query_vectors)} 个向量)")
    fvecs_write(query_file, query_vectors)
    
    print(f"保存 base 向量到: {base_file} ({len(base_vectors)} 个向量)")
    fvecs_write(base_file, base_vectors)
    
    return query_vectors, base_vectors


def create_index(base_vectors, use_gpu=False):
    """
    创建索引用于 groundtruth 生成
    
    Args:
        base_vectors: base 向量数组
        use_gpu: 是否使用 GPU
    
    Returns:
        faiss.Index: 创建的索引
    """
    dimension = base_vectors.shape[1]
    
    if use_gpu:
        # 检查 GPU 可用性
        if not faiss.get_num_gpus():
            raise RuntimeError("未检测到可用的 GPU")
        
        print(f"使用 GPU 创建索引 (维度: {dimension})")
        # 创建 GPU 索引
        res = faiss.StandardGpuResources()
        index_cpu = faiss.IndexFlatL2(dimension)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        print(f"使用 CPU 创建索引 (维度: {dimension})")
        index = faiss.IndexFlatL2(dimension)
    
    # 添加 base 向量到索引
    print(f"添加 {len(base_vectors)} 个向量到索引...")
    index.add(base_vectors.astype(np.float32))
    
    return index


def generate_groundtruth_batch(query_vectors, index, topk, batch_size=1000):
    """
    分批生成 groundtruth
    
    Args:
        query_vectors: query 向量数组
        index: faiss 索引
        topk: 每个 query 返回的最近邻数量
        batch_size: 批处理大小
    
    Returns:
        np.ndarray: groundtruth 索引数组
    """
    num_queries = len(query_vectors)
    all_indices = []
    
    print(f"正在生成 groundtruth (topk={topk}, batch_size={batch_size})...")
    
    # 分批处理
    for start_idx in tqdm(range(0, num_queries, batch_size), desc="处理批次"):
        end_idx = min(start_idx + batch_size, num_queries)
        batch_queries = query_vectors[start_idx:end_idx]
        
        # 搜索最近邻
        distances, indices = index.search(batch_queries.astype(np.float32), topk)
        all_indices.append(indices)
    
    # 合并所有批次的结果
    groundtruth = np.vstack(all_indices)
    return groundtruth


def estimate_memory_usage(num_base, num_queries, dimension, topk, use_gpu=False):
    """
    估算内存使用量
    
    Args:
        num_base: base 向量数量
        num_queries: query 向量数量
        dimension: 向量维度
        topk: 返回的最近邻数量
        use_gpu: 是否使用 GPU
    
    Returns:
        dict: 内存使用估算
    """
    # 基础数据内存 (float32)
    base_memory = num_base * dimension * 4  # bytes
    query_memory = num_queries * dimension * 4  # bytes
    
    # 索引内存 (大约等于基础数据)
    index_memory = base_memory
    
    # 结果内存
    result_memory = num_queries * topk * 4  # int32
    
    total_memory = base_memory + query_memory + index_memory + result_memory
    
    return {
        'base_vectors': base_memory / (1024**3),  # GB
        'query_vectors': query_memory / (1024**3),  # GB
        'index': index_memory / (1024**3),  # GB
        'results': result_memory / (1024**3),  # GB
        'total': total_memory / (1024**3),  # GB
        'device': 'GPU' if use_gpu else 'CPU'
    }


def suggest_batch_size(num_queries, available_memory_gb=8.0):
    """
    根据可用内存建议批处理大小
    
    Args:
        num_queries: query 向量数量
        available_memory_gb: 可用内存 (GB)
    
    Returns:
        int: 建议的批处理大小
    """
    # 保守估计，假设每个 query 需要 1KB 内存用于中间计算
    memory_per_query = 0.001  # GB
    max_batch = int(available_memory_gb / memory_per_query)
    
    # 限制在合理范围内
    suggested_batch = min(max_batch, max(100, num_queries // 10))
    return suggested_batch


def main():
    parser = argparse.ArgumentParser(
        description="将 .fvecs 文件分割成 query/base 并生成 groundtruth (支持 GPU 加速)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # CPU 模式
  python generate_dataset.py -i data/sift.fvecs -q 1000 -k 100 -o data/sift
  
  # GPU 模式，自定义批处理大小
  python generate_dataset.py -i data/sift.fvecs -q 1000 -k 100 -o data/sift --gpu --batch-size 500
  
  这将会生成:
  - data/sift_query.fvecs (1000 个 query 向量)
  - data/sift_base.fvecs (剩余的 base 向量)
  - data/sift_groundtruth.ivecs (每个 query 的前 100 个最近邻)
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='输入的 .fvecs 文件路径')
    parser.add_argument('-q', '--queries', type=int, required=True,
                        help='query 集合的向量数量')
    parser.add_argument('-k', '--topk', type=int, default=100,
                        help='每个 query 的最近邻数量 (默认: 100)')
    parser.add_argument('-o', '--output', required=True,
                        help='输出文件前缀 (不含扩展名)')
    parser.add_argument('--gpu', action='store_true',
                        help='使用 GPU 加速 groundtruth 生成')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='批处理大小 (0 表示自动选择)')
    parser.add_argument('--memory-limit', type=float, default=8.0,
                        help='内存限制 (GB，用于自动选择批处理大小，默认: 8.0)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return 1
    
    # 检查输入文件扩展名
    if not args.input.endswith('.fvecs'):
        print(f"错误: 输入文件必须是 .fvecs 格式")
        return 1
    
    try:
        # 分割数据集
        query_vectors, base_vectors = split_dataset(
            args.input, args.queries, args.output
        )
        
        # 估算内存使用
        memory_info = estimate_memory_usage(
            len(base_vectors), len(query_vectors), 
            base_vectors.shape[1], args.topk, args.gpu
        )
        
        print(f"\n内存使用估算 ({memory_info['device']}):")
        print(f"  Base 向量: {memory_info['base_vectors']:.2f} GB")
        print(f"  Query 向量: {memory_info['query_vectors']:.2f} GB")
        print(f"  索引: {memory_info['index']:.2f} GB")
        print(f"  结果: {memory_info['results']:.2f} GB")
        print(f"  总计: {memory_info['total']:.2f} GB")
        
        # 确定批处理大小
        if args.batch_size == 0:
            batch_size = suggest_batch_size(len(query_vectors), args.memory_limit)
            print(f"\n自动选择批处理大小: {batch_size}")
        else:
            batch_size = args.batch_size
            print(f"\n使用指定批处理大小: {batch_size}")
        
        # 创建索引
        index = create_index(base_vectors, args.gpu)
        
        # 生成 groundtruth
        groundtruth_file = f"{args.output}_groundtruth.ivecs"
        groundtruth = generate_groundtruth_batch(
            query_vectors, index, args.topk, batch_size
        )
        
        # 保存 groundtruth
        print(f"保存 groundtruth 到: {groundtruth_file}")
        ivecs_write(groundtruth_file, groundtruth.astype(np.int32))
        
        print("\n数据集生成完成!")
        print(f"Query 向量数量: {len(query_vectors)}")
        print(f"Base 向量数量: {len(base_vectors)}")
        print(f"Groundtruth topk: {args.topk}")
        print(f"使用设备: {'GPU' if args.gpu else 'CPU'}")
        print(f"批处理大小: {batch_size}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())