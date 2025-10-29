#!/usr/bin/env python3
"""
数据集生成工具

将单个 .fvecs 文件分割成 query 和 base 两部分，并生成对应的 groundtruth 文件。
"""

import argparse
import os
import numpy as np
import faiss
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


def generate_groundtruth(query_vectors, base_vectors, topk, output_file):
    """
    使用暴力搜索生成 groundtruth
    
    Args:
        query_vectors: query 向量数组
        base_vectors: base 向量数组
        topk: 每个 query 返回的最近邻数量
        output_file: groundtruth 输出文件路径
    """
    print(f"正在生成 groundtruth (topk={topk})...")
    
    # 创建 Flat 索引进行暴力搜索
    dimension = base_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # 添加 base 向量到索引
    index.add(base_vectors.astype(np.float32))
    
    # 搜索最近邻
    distances, indices = index.search(query_vectors.astype(np.float32), topk)
    
    # 保存 groundtruth
    print(f"保存 groundtruth 到: {output_file}")
    ivecs_write(output_file, indices.astype(np.int32))
    
    return indices


def main():
    parser = argparse.ArgumentParser(
        description="将 .fvecs 文件分割成 query/base 并生成 groundtruth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python generate_dataset.py -i data/sift.fvecs -q 1000 -k 100 -o data/sift
  
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
        
        # 生成 groundtruth
        groundtruth_file = f"{args.output}_groundtruth.ivecs"
        generate_groundtruth(
            query_vectors, base_vectors, args.topk, groundtruth_file
        )
        
        print("\n数据集生成完成!")
        print(f"Query 向量数量: {len(query_vectors)}")
        print(f"Base 向量数量: {len(base_vectors)}")
        print(f"Groundtruth topk: {args.topk}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())