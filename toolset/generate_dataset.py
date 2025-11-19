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
from faiss_benchmark.utils import (
    fvecs_read, fvecs_write, ivecs_write, 
    get_fvecs_info, fvecs_read_range, fvecs_write_streaming
)


def split_dataset(input_file, num_queries, output_prefix, chunk_size=50000):
    """
    将输入的 .fvecs 文件分割成 query 和 base 两部分（流式处理，内存友好）

    Args:
        input_file: 输入的 .fvecs 文件路径
        num_queries: query 集合的向量数量
        output_prefix: 输出文件的前缀（不含扩展名）
        chunk_size: 每次处理的向量数量（默认50000，约200MB内存）

    Returns:
        tuple: (query_vectors, base_vectors) 分别为 query 和 base 向量数组
    """
    print(f"正在分析数据文件: {input_file}")
    
    # 获取文件信息，不加载数据
    total_vectors, dimension = get_fvecs_info(input_file)
    
    if total_vectors < num_queries:
        raise ValueError(
            f"数据集总数量 ({total_vectors}) 小于请求的 query 数量 ({num_queries})"
        )

    num_base = total_vectors - num_queries
    
    print(f"数据集信息: {total_vectors} 个向量, {dimension} 维")
    print(f"将分割为: {num_queries} 个 query 向量, {num_base} 个 base 向量")
    
    # 输出文件路径
    query_file = f"{output_prefix}_query.fvecs"
    base_file = f"{output_prefix}_base.fvecs"

    # 流式处理 query 向量
    print(f"正在保存 query 向量到: {query_file}")
    
    def query_generator():
        remaining = num_queries
        current_idx = 0
        
        while remaining > 0:
            batch_size = min(chunk_size, remaining)
            vectors = fvecs_read_range(input_file, current_idx, batch_size)
            yield vectors
            
            current_idx += batch_size
            remaining -= batch_size
    
    fvecs_write_streaming(query_file, query_generator(), num_queries)
    
    # 流式处理 base 向量
    print(f"正在保存 base 向量到: {base_file}")
    
    def base_generator():
        remaining = num_base
        current_idx = num_queries
        
        while remaining > 0:
            batch_size = min(chunk_size, remaining)
            vectors = fvecs_read_range(input_file, current_idx, batch_size)
            yield vectors
            
            current_idx += batch_size
            remaining -= batch_size
    
    fvecs_write_streaming(base_file, base_generator(), num_base)
    
    print("数据分割完成!")
    
    # 为了保持向后兼容，返回小批量的数据用于后续处理
    # 只加载少量数据到内存，而不是全部
    sample_size = min(1000, num_queries)
    query_sample = fvecs_read_range(input_file, 0, sample_size)
    
    sample_size = min(1000, num_base)
    base_sample = fvecs_read_range(input_file, num_queries, sample_size)
    
    return query_sample, base_sample


def check_memory_feasibility(num_vectors, dimension, gpu_memory_gb=10):
    """
    检查在给定 GPU 内存下是否可以建立索引
    
    Args:
        num_vectors: 向量数量
        dimension: 向量维度
        gpu_memory_gb: GPU 内存大小 (GB)
        
    Returns:
        dict: 包含可行性分析和建议的字典
    """
    bytes_per_float = 4
    
    # 计算所需内存
    data_memory_gb = (num_vectors * dimension * bytes_per_float) / (1024**3)
    index_memory_gb = data_memory_gb * 1.1  # 10% 开销
    
    # 计算分块方案
    available_memory_gb = gpu_memory_gb - 1  # 预留 1GB
    max_vectors_per_chunk = int((available_memory_gb * (1024**3)) / (dimension * bytes_per_float * 1.1))
    num_chunks = (num_vectors + max_vectors_per_chunk - 1) // max_vectors_per_chunk
    
    return {
        'total_memory_required_gb': index_memory_gb,
        'gpu_memory_gb': gpu_memory_gb,
        'feasible': index_memory_gb <= gpu_memory_gb,
        'max_vectors_per_chunk': max_vectors_per_chunk,
        'num_chunks_needed': num_chunks,
        'chunk_memory_gb': (max_vectors_per_chunk * dimension * bytes_per_float * 1.1) / (1024**3)
    }


def calculate_dynamic_batch_size(current_vectors, total_vectors, dimension, gpu_memory_gb, initial_batch_size=50000):
    """
    根据当前索引大小动态计算批处理大小
    
    Args:
        current_vectors: 当前已添加的向量数
        total_vectors: 总向量数
        dimension: 向量维度
        gpu_memory_gb: GPU 内存大小
        initial_batch_size: 初始批处理大小
        
    Returns:
        int: 建议的批处理大小
    """
    bytes_per_float = 4
    
    # 计算当前索引占用的内存
    current_index_memory_gb = (current_vectors * dimension * bytes_per_float * 2.2) / (1024**3)
    
    # 预留内存（系统 + 缓冲区）
    reserved_memory_gb = 1.5
    
    # 计算可用内存
    available_memory_gb = gpu_memory_gb - current_index_memory_gb - reserved_memory_gb
    
    if available_memory_gb <= 0:
        # 内存不足，使用最小批大小
        return min(1000, initial_batch_size)
    
    # 计算最大安全批大小（考虑 1.5x 开销）
    max_safe_batch_size = int(available_memory_gb * 1024**3 / (dimension * bytes_per_float * 1.5))
    
    # 限制在合理范围内
    min_batch_size = 1000
    max_batch_size = min(100000, initial_batch_size * 2)
    
    optimal_batch_size = max(min_batch_size, min(max_safe_batch_size, max_batch_size))
    
    return optimal_batch_size


def create_index(base_file_or_vectors, use_gpu=False, chunk_size=50000, gpu_memory_gb=10, ngpu: int | None = None, gpu_shard: bool = True):
    """
    创建索引用于 groundtruth 生成（支持流式加载和内存检查）

    Args:
        base_file_or_vectors: base 向量文件路径或向量数组
        use_gpu: 是否使用 GPU
        chunk_size: 流式加载时的批处理大小
        gpu_memory_gb: GPU 内存大小 (GB)，用于内存可行性检查

    Returns:
        faiss.Index: 创建的索引
    """
    # 判断输入类型
    if isinstance(base_file_or_vectors, str):
        # 文件路径，使用流式加载
        base_file = base_file_or_vectors
        total_vectors, dimension = get_fvecs_info(base_file)
        
        print(f"从文件流式加载 base 向量: {base_file}")
        print(f"向量信息: {total_vectors} 个向量, {dimension} 维")
        
        # 如果使用 GPU，检查内存可行性
        if use_gpu:
            memory_check = check_memory_feasibility(total_vectors, dimension, gpu_memory_gb)
            print(f"\n🔍 GPU 内存可行性检查:")
            print(f"  所需内存: {memory_check['total_memory_required_gb']:.2f} GB")
            print(f"  可用内存: {memory_check['gpu_memory_gb']} GB")
            
            if not memory_check['feasible']:
                print(f"  ❌ 内存不足！超出 {memory_check['total_memory_required_gb'] - memory_check['gpu_memory_gb']:.2f} GB")
                print(f"\n💡 建议的分块方案:")
                print(f"  每块最大向量数: {memory_check['max_vectors_per_chunk']:,}")
                print(f"  需要的块数: {memory_check['num_chunks_needed']}")
                print(f"  每块内存占用: {memory_check['chunk_memory_gb']:.2f} GB")
                print(f"\n⚠️  警告: 当前配置可能导致 GPU 内存溢出")
                print(f"     建议使用 CPU 模式或减少数据量")
            else:
                print(f"  ✅ 内存充足，可以使用 GPU 模式")
        
        print()
        
        # 确保 dimension 是正确的整数类型
        dimension = int(dimension)
        
        if use_gpu:
            # 检查 GPU 可用性
            n_available = faiss.get_num_gpus()
            if n_available <= 0:
                raise RuntimeError("未检测到可用的 GPU")
            # 选择多 GPU 数量
            if ngpu is None:
                ngpu = n_available
            ngpu = max(1, min(ngpu, n_available))

            print(f"使用 {ngpu} 个 GPU 创建索引 (维度: {dimension}) | shard={gpu_shard}")
            # 创建 CPU 基索引并克隆到所有可见 GPU
            index_cpu = faiss.IndexFlatL2(int(dimension))
            co = faiss.GpuMultipleClonerOptions()
            co.shard = bool(gpu_shard)
            # 使用所有可见 GPU；如需指定 GPU，请设置 CUDA_VISIBLE_DEVICES 环境变量
            index = faiss.index_cpu_to_all_gpus(index_cpu, co)
        else:
            print(f"使用 CPU 创建索引 (维度: {dimension})")
            index = faiss.IndexFlatL2(int(dimension))

        # 流式添加向量到索引
        print(f"正在流式添加 {total_vectors} 个向量到索引...")
        if use_gpu:
            # 取消动态批大小调整，始终使用用户提供的 batch_size
            print(f"使用固定批处理大小（GPU 模式）：{chunk_size}")
        
        current_idx = 0
        remaining = total_vectors
        batch_count = 0
        
        with tqdm(total=total_vectors, desc="添加向量") as pbar:
            while remaining > 0:
                # 取消动态批次调整，始终使用配置的 chunk_size
                batch_size = min(chunk_size, remaining)
                
                vectors = fvecs_read_range(base_file, current_idx, batch_size)
                
                # 添加到索引
                index.add(vectors.astype(np.float32))
                
                current_idx += batch_size
                remaining -= batch_size
                batch_count += 1
                pbar.update(batch_size)
                
                # 清理内存
                del vectors
        
    else:
        # 向量数组，保持原有逻辑
        base_vectors = base_file_or_vectors
        dimension = int(base_vectors.shape[1])

        if use_gpu:
            # 检查 GPU 可用性
            n_available = faiss.get_num_gpus()
            if n_available <= 0:
                raise RuntimeError("未检测到可用的 GPU")
            if ngpu is None:
                ngpu = n_available
            ngpu = max(1, min(ngpu, n_available))

            print(f"使用 {ngpu} 个 GPU 创建索引 (维度: {dimension}) | shard={gpu_shard}")
            index_cpu = faiss.IndexFlatL2(int(dimension))
            co = faiss.GpuMultipleClonerOptions()
            co.shard = bool(gpu_shard)
            index = faiss.index_cpu_to_all_gpus(index_cpu, co)
        else:
            print(f"使用 CPU 创建索引 (维度: {dimension})")
            index = faiss.IndexFlatL2(int(dimension))

        # 添加 base 向量到索引
        print(f"添加 {len(base_vectors)} 个向量到索引...")
        index.add(base_vectors.astype(np.float32))

    return index





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
        "base_vectors": base_memory / (1024**3),  # GB
        "query_vectors": query_memory / (1024**3),  # GB
        "index": index_memory / (1024**3),  # GB
        "results": result_memory / (1024**3),  # GB
        "total": total_memory / (1024**3),  # GB
        "device": "GPU" if use_gpu else "CPU",
    }


def suggest_batch_size(num_queries, available_memory_gb=16.0):
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
  # CPU 模式（输入为数据集目录路径，目录名即数据集名）
  python generate_dataset.py -i data/sift -q 1000 -k 100

  # GPU 模式，自动选择批处理大小
  python generate_dataset.py -i data/sift -q 1000 -k 100 --gpu

  # GPU 模式，自定义批处理大小
  python generate_dataset.py -i data/sift -q 1000 -k 100 --gpu --batch-size 500

  # GPU 模式，指定 GPU 内存大小（用于可行性检查与分块建议）
  python generate_dataset.py -i data/large -q 1000 -k 100 --gpu --gpu-memory 24

  这将会在数据集目录下生成:
  - data/sift/sift_query.fvecs (1000 个 query 向量)
  - data/sift/sift_base.fvecs (剩余的 base 向量)
  - data/sift/sift_groundtruth.ivecs (每个 query 的前 100 个最近邻)

  注意:
  - 输入参数 -i/--input 需要提供数据集目录路径，目录下需包含同名 .fvecs 文件，例如 data/sift/sift.fvecs。
  - 使用 --gpu 时，会进行 GPU 内存可行性检查；若不足将提示分块建议。
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="输入的数据集目录路径")
    parser.add_argument(
        "-q", "--queries", type=int, required=True, help="query 集合的向量数量"
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        default=100,
        help="每个 query 的最近邻数量 (默认: 100)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="使用 GPU 加速 groundtruth 生成"
    )
    parser.add_argument(
        "--ngpu", type=int, default=0, help="使用的 GPU 数量 (默认 0 表示使用所有可用 GPU)"
    )
    parser.add_argument(
        "--gpu-shard", action="store_true", default=True, help="启用分片模式（跨 GPU 分片而非复制，默认开启）"
    )
    parser.add_argument(
        "--no-gpu-shard", dest="gpu_shard", action="store_false", help="关闭分片模式，改为在每个 GPU 复制完整索引"
    )
    parser.add_argument(
        "--batch-size", type=int, default=0, help="批处理大小 (0 表示自动选择)"
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=16.0,
        help="内存限制 (GB，用于自动选择批处理大小，默认: 16.0)",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=10.0,
        help="GPU 内存大小 (GB，用于内存可行性检查，默认: 10.0)",
    )

    args = parser.parse_args()
    
    data_base_dir = args.input
    if not os.path.exists(data_base_dir):
        print(f"错误: 数据目录 {data_base_dir} 不存在")
        return 1
    data_name = os.path.basename(data_base_dir)
    
    # 检查输入文件是否存在
    input_file = os.path.join(data_base_dir, f"{data_name}.fvecs")
    if not os.path.exists(input_file):
        print(f"错误: 输入数据 {input_file} 不存在")
        return 1

    # 检查输入文件扩展名
    if not input_file.endswith(".fvecs"):
        print(f"错误: 输入文件必须是 .fvecs 格式")
        return 1

    try:
        # 获取数据集信息（不加载数据）
        total_vectors, dimension = get_fvecs_info(input_file)
        num_base = total_vectors - args.queries
        
        print(f"\n数据集信息:")
        print(f"  总向量数: {total_vectors}")
        print(f"  维度: {dimension}")
        print(f"  Query 向量数: {args.queries}")
        print(f"  Base 向量数: {num_base}")

        # 分割数据集（流式处理，内存友好）
        output_prefix = os.path.join(data_base_dir, data_name)
        query_sample, base_sample = split_dataset(
            input_file, args.queries, output_prefix
        )

        # 估算内存使用（基于实际数据量）
        memory_info = estimate_memory_usage(
            num_base,
            args.queries,
            dimension,
            args.topk,
            args.gpu,
        )

        print(f"\n内存使用估算 ({memory_info['device']}):")
        print(f"  Base 向量: {memory_info['base_vectors']:.2f} GB")
        print(f"  Query 向量: {memory_info['query_vectors']:.2f} GB")
        print(f"  索引: {memory_info['index']:.2f} GB")
        print(f"  结果: {memory_info['results']:.2f} GB")
        print(f"  总计: {memory_info['total']:.2f} GB")

        # 确定批处理大小
        if args.batch_size == 0:
            batch_size = suggest_batch_size(args.queries, args.memory_limit)
            print(f"\n自动选择批处理大小: {batch_size}")
        else:
            batch_size = args.batch_size
            print(f"\n使用指定批处理大小: {batch_size}")

        # 清理样本数据，释放内存
        del query_sample, base_sample
        
        # 创建索引（使用流式加载）
        base_file = f"{output_prefix}_base.fvecs"
        ngpu_val = None if args.ngpu == 0 else int(args.ngpu)
        index = create_index(base_file, args.gpu, gpu_memory_gb=args.gpu_memory, ngpu=ngpu_val, gpu_shard=bool(args.gpu_shard))

        # 生成 groundtruth（使用流式加载 query 向量）
        query_file = f"{output_prefix}_query.fvecs"
        groundtruth_file = f"{output_prefix}_groundtruth.ivecs"
        
        print(f"\n正在生成 groundtruth...")
        
        # 流式处理 query 向量生成 groundtruth
        all_groundtruth = []
        current_idx = 0
        remaining_queries = args.queries
        
        with tqdm(total=args.queries, desc="生成 groundtruth") as pbar:
            while remaining_queries > 0:
                current_batch_size = min(batch_size, remaining_queries)
                
                # 读取当前批次的 query 向量
                query_batch = fvecs_read_range(query_file, current_idx, current_batch_size)
                
                # 搜索最近邻
                _, batch_groundtruth = index.search(query_batch.astype(np.float32), args.topk)
                all_groundtruth.append(batch_groundtruth)
                
                current_idx += current_batch_size
                remaining_queries -= current_batch_size
                pbar.update(current_batch_size)
                
                # 清理当前批次数据
                del query_batch
        
        # 合并所有 groundtruth
        groundtruth = np.vstack(all_groundtruth)
        del all_groundtruth  # 清理中间数据

        # 保存 groundtruth
        print(f"保存 groundtruth 到: {groundtruth_file}")
        ivecs_write(groundtruth_file, groundtruth.astype(np.int32))

        print("\n数据集生成完成!")
        print(f"Query 向量数量: {args.queries}")
        print(f"Base 向量数量: {num_base}")
        print(f"Groundtruth topk: {args.topk}")
        print(f"使用设备: {'GPU' if args.gpu else 'CPU'}")
        print(f"批处理大小: {batch_size}")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
