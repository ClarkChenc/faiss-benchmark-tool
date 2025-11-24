#!/usr/bin/env python3
"""
从已分割的数据集目录生成 groundtruth 的独立脚本。

输入为数据集目录路径，脚本会自动读取该目录下的：
- <name>_base.fvecs
- <name>_query.fvecs

并生成：
- <name>_groundtruth.ivecs

生成方式与现有 generate_dataset.py 中逻辑保持一致，支持 CPU/GPU 两种路径，
并支持批处理以降低内存占用。
"""

import argparse
import os
import sys
import numpy as np
import faiss
from tqdm import tqdm

# 允许导入 faiss_benchmark.utils 和复用 generate_dataset 中的索引创建/估算方法
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from faiss_benchmark.utils import (
    ivecs_write,
    get_fvecs_info,
    fvecs_read_range,
)

try:
    # 复用已有的索引创建和内存估算逻辑
    from toolset.generate_dataset import create_index, estimate_memory_usage, suggest_batch_size
except Exception:
    # 兜底：简单估算函数与最小化索引创建（CPU-only），避免导入失败导致脚本不可用
    def estimate_memory_usage(num_base, num_queries, dimension, topk, use_gpu=False):
        base_memory = num_base * dimension * 4
        query_memory = num_queries * dimension * 4
        index_memory = base_memory
        result_memory = num_queries * topk * 4
        total_memory = base_memory + query_memory + index_memory + result_memory
        return {
            "base_vectors": base_memory / (1024**3),
            "query_vectors": query_memory / (1024**3),
            "index": index_memory / (1024**3),
            "results": result_memory / (1024**3),
            "total": total_memory / (1024**3),
            "device": "GPU" if use_gpu else "CPU",
        }

    def suggest_batch_size(num_queries, available_memory_gb=16.0):
        memory_per_query = 0.001  # GB
        max_batch = int(available_memory_gb / memory_per_query)
        suggested_batch = min(max_batch, max(100, num_queries // 10))
        return suggested_batch

    def create_index(base_file_or_vectors, use_gpu=False, **kwargs):
        # 最小化 CPU 索引创建（兜底）：直接全量加载（不推荐，但避免导入失败）
        if isinstance(base_file_or_vectors, str):
            total, dim = get_fvecs_info(base_file_or_vectors)
            # 流式加载以减少内存压力
            idx = faiss.IndexFlatL2(int(dim))
            processed = 0
            with tqdm(total=total, desc="添加向量(兜底)") as pbar:
                while processed < total:
                    bs = min(50000, total - processed)
                    xb = fvecs_read_range(base_file_or_vectors, processed, bs)
                    idx.add(xb.astype(np.float32))
                    processed += bs
                    pbar.update(bs)
                    del xb
            return idx
        else:
            xb = np.asarray(base_file_or_vectors, dtype=np.float32)
            return faiss.IndexFlatL2(int(xb.shape[1]))


def main():
    parser = argparse.ArgumentParser(
        description="从数据集目录生成 groundtruth (支持 GPU 加速)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # CPU 模式
  python generate_groundtruth.py -i data/sift -k 100

  # GPU 模式（自动选择批处理大小）
  python generate_groundtruth.py -i data/sift -k 100 --gpu

  # GPU 模式（自定义批处理大小）
  python generate_groundtruth.py -i data/sift -k 100 --gpu --batch-size 500

生成的文件：
  - data/sift/sift_groundtruth.ivecs
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="数据集目录路径")
    parser.add_argument("-k", "--topk", type=int, default=100, help="每个 query 的最近邻数量")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 加速生成")
    parser.add_argument("--ngpu", type=int, default=0, help="GPU 数量 (0 表示使用所有可用 GPU)")
    parser.add_argument("--gpu-shard", action="store_true", default=True, help="启用 GPU 分片模式 (默认开启)")
    parser.add_argument("--no-gpu-shard", dest="gpu_shard", action="store_false", help="关闭分片模式，改为复制索引到每个 GPU")
    parser.add_argument("--batch-size", type=int, default=0, help="生成时的批处理大小 (0 表示自动选择)")
    parser.add_argument("--memory-limit", type=float, default=16.0, help="内存限制 (GB，用于自动选择批处理大小)")
    parser.add_argument("--gpu-memory", type=float, default=10.0, help="GPU 内存大小 (GB，用于可行性检查)")

    args = parser.parse_args()

    data_dir = args.input
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        return 1
    name = os.path.basename(data_dir)

    base_file = os.path.join(data_dir, f"{name}_base.fvecs")
    query_file = os.path.join(data_dir, f"{name}_query.fvecs")
    gt_file = os.path.join(data_dir, f"{name}_groundtruth.ivecs")

    if not os.path.exists(base_file):
        print(f"错误: 未找到 base 向量文件: {base_file}")
        return 1
    if not os.path.exists(query_file):
        print(f"错误: 未找到 query 向量文件: {query_file}")
        return 1

    try:
        # 获取统计信息
        num_base, dim = get_fvecs_info(base_file)
        num_queries, dim_q = get_fvecs_info(query_file)
        if dim != dim_q:
            print(f"警告: base({dim}) 与 query({dim_q}) 维度不一致，以 base 维度为准")

        print("\n数据集信息:")
        print(f"  Base 向量数: {num_base}")
        print(f"  Query 向量数: {num_queries}")
        print(f"  维度: {dim}")

        # 估算内存使用
        mem = estimate_memory_usage(num_base, num_queries, dim, args.topk, args.gpu)
        print(f"\n内存使用估算 ({mem['device']}):")
        print(f"  Base: {mem['base_vectors']:.2f} GB")
        print(f"  Query: {mem['query_vectors']:.2f} GB")
        print(f"  索引: {mem['index']:.2f} GB")
        print(f"  结果: {mem['results']:.2f} GB")
        print(f"  总计: {mem['total']:.2f} GB")

        # 选择批处理大小
        if args.batch_size == 0:
            batch_size = suggest_batch_size(num_queries, args.memory_limit)
            print(f"\n自动选择批处理大小: {batch_size}")
        else:
            batch_size = int(args.batch_size)
            print(f"\n使用指定批处理大小: {batch_size}")

        # 创建索引用于 groundtruth 生成（流式加载 base）
        ngpu_val = None if args.ngpu == 0 else int(args.ngpu)
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        device_ids = None
        if vis:
            try:
                device_ids = [int(x) for x in vis.split(',') if x.strip()!='']
            except Exception:
                device_ids = None
        index = create_index(base_file, args.gpu, gpu_memory_gb=args.gpu_memory, ngpu=ngpu_val, gpu_shard=bool(args.gpu_shard), device_ids=device_ids)

        # 生成 groundtruth（流式处理 query）
        print("\n正在生成 groundtruth...")
        all_gt = []
        processed = 0
        remaining = num_queries
        with tqdm(total=num_queries, desc="生成 groundtruth") as pbar:
            while remaining > 0:
                bs = min(batch_size, remaining)
                xq = fvecs_read_range(query_file, processed, bs)
                _, I = index.search(xq.astype(np.float32), args.topk)
                all_gt.append(I)
                processed += bs
                remaining -= bs
                pbar.update(bs)
                del xq

        groundtruth = np.vstack(all_gt)
        del all_gt

        print(f"保存 groundtruth 到: {gt_file}")
        ivecs_write(gt_file, groundtruth.astype(np.int32))

        print("\n生成完成！")
        print(f"Query 数量: {num_queries}")
        print(f"Base 数量: {num_base}")
        print(f"TopK: {args.topk}")
        print(f"设备: {'GPU' if args.gpu else 'CPU'}")
        print(f"批处理大小: {batch_size}")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

