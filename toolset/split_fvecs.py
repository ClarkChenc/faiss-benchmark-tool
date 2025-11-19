#!/usr/bin/env python3
"""
.fvecs 数据分割工具

将大型 .fvecs 文件分割成指定大小的多个小文件，支持内存优化的流式处理。
适用于处理大规模向量数据集，避免内存溢出问题。
"""

import argparse
import os
import math
import numpy as np
from tqdm import tqdm
from ../faiss_benchmark.utils import (
    get_fvecs_info, fvecs_read_range, fvecs_write_streaming
)


def split_fvecs_file(input_file, output_dir, chunk_size, output_prefix=None, start_index=0):
    """
    将 .fvecs 文件分割成指定大小的多个文件
    
    Args:
        input_file: 输入的 .fvecs 文件路径
        output_dir: 输出目录
        chunk_size: 每个分割文件的向量数量
        output_prefix: 输出文件名前缀（默认使用输入文件名）
        start_index: 起始索引（默认从0开始）
    
    Returns:
        list: 生成的文件路径列表
    """
    print(f"🔍 正在分析输入文件: {input_file}")
    
    # 获取文件信息
    total_vectors, dimension = get_fvecs_info(input_file)
    
    print(f"📊 文件信息:")
    print(f"  总向量数: {total_vectors:,}")
    print(f"  向量维度: {dimension}")
    print(f"  文件大小: {os.path.getsize(input_file) / (1024**3):.2f} GB")
    
    # 计算分割参数
    if start_index >= total_vectors:
        raise ValueError(f"起始索引 {start_index} 超出文件范围 (0-{total_vectors-1})")
    
    available_vectors = total_vectors - start_index
    num_chunks = math.ceil(available_vectors / chunk_size)
    
    print(f"📦 分割计划:")
    print(f"  起始索引: {start_index}")
    print(f"  可用向量: {available_vectors:,}")
    print(f"  每块大小: {chunk_size:,}")
    print(f"  分割块数: {num_chunks}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定输出文件前缀
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(input_file))[0]
    
    # 分割文件
    output_files = []
    current_index = start_index
    
    print(f"🚀 开始分割...")
    
    for chunk_idx in tqdm(range(num_chunks), desc="分割进度"):
        # 计算当前块的大小
        remaining_vectors = available_vectors - (chunk_idx * chunk_size)
        current_chunk_size = min(chunk_size, remaining_vectors)
        
        # 生成输出文件名
        output_filename = f"{output_prefix}_part_{chunk_idx+1:04d}.fvecs"
        output_path = os.path.join(output_dir, output_filename)
        
        # 创建向量生成器
        def vector_generator():
            # 使用较小的批次读取，避免内存问题
            batch_size = min(50000, current_chunk_size)
            read_index = current_index
            remaining = current_chunk_size
            
            while remaining > 0:
                actual_batch_size = min(batch_size, remaining)
                vectors = fvecs_read_range(input_file, read_index, actual_batch_size)
                yield vectors
                
                read_index += actual_batch_size
                remaining -= actual_batch_size
        
        # 写入文件
        fvecs_write_streaming(output_path, vector_generator(), current_chunk_size)
        
        output_files.append(output_path)
        current_index += current_chunk_size
        
        # 显示当前块信息
        tqdm.write(f"  ✅ 生成: {output_filename} ({current_chunk_size:,} 向量)")
    
    print(f"🎉 分割完成!")
    print(f"  生成文件数: {len(output_files)}")
    print(f"  输出目录: {output_dir}")
    
    # 验证分割结果
    print(f"🔍 验证分割结果...")
    total_output_vectors = 0
    
    for i, output_file in enumerate(output_files):
        vectors, dim = get_fvecs_info(output_file)
        total_output_vectors += vectors
        
        if dim != dimension:
            print(f"  ⚠️ 警告: {output_file} 维度不匹配 ({dim} vs {dimension})")
        
        print(f"  📄 {os.path.basename(output_file)}: {vectors:,} 向量")
    
    if total_output_vectors == available_vectors:
        print(f"  ✅ 验证通过: 总向量数匹配 ({total_output_vectors:,})")
    else:
        print(f"  ❌ 验证失败: 向量数不匹配 ({total_output_vectors:,} vs {available_vectors:,})")
    
    return output_files


def estimate_memory_usage(total_vectors, dimension, chunk_size):
    """
    估算内存使用量
    
    Args:
        total_vectors: 总向量数
        dimension: 向量维度
        chunk_size: 分割大小
    
    Returns:
        dict: 内存使用估算信息
    """
    # 每个向量的字节数 (float32)
    bytes_per_vector = dimension * 4
    
    # 单个分割文件的内存需求
    chunk_memory_mb = (chunk_size * bytes_per_vector) / (1024 * 1024)
    
    # 读取缓冲区内存 (假设50k向量的批次)
    buffer_size = min(50000, chunk_size)
    buffer_memory_mb = (buffer_size * bytes_per_vector) / (1024 * 1024)
    
    # 总内存估算 (包括一些开销)
    total_memory_mb = chunk_memory_mb + buffer_memory_mb + 100  # 100MB 开销
    
    return {
        'chunk_memory_mb': chunk_memory_mb,
        'buffer_memory_mb': buffer_memory_mb,
        'total_memory_mb': total_memory_mb,
        'chunk_memory_gb': chunk_memory_mb / 1024,
        'total_memory_gb': total_memory_mb / 1024
    }


def suggest_chunk_size(total_vectors, dimension, max_memory_gb=4.0):
    """
    根据内存限制建议合适的分割大小
    
    Args:
        total_vectors: 总向量数
        dimension: 向量维度
        max_memory_gb: 最大内存限制 (GB)
    
    Returns:
        int: 建议的分割大小
    """
    bytes_per_vector = dimension * 4
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    
    # 预留一些内存给缓冲区和开销
    available_memory_bytes = max_memory_bytes * 0.8
    
    suggested_chunk_size = int(available_memory_bytes / bytes_per_vector)
    
    # 确保不超过总向量数
    suggested_chunk_size = min(suggested_chunk_size, total_vectors)
    
    # 确保至少有1000个向量
    suggested_chunk_size = max(suggested_chunk_size, 1000)
    
    return suggested_chunk_size


def main():
    parser = argparse.ArgumentParser(
        description=".fvecs 数据分割工具 - 将大型向量文件分割成指定大小的多个文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本分割：将文件分割成每个100万向量的块
  python split_fvecs.py -i data/large.fvecs -o data/splits -s 1000000
  
  # 自定义输出前缀
  python split_fvecs.py -i data/sift.fvecs -o data/splits -s 500000 -p sift_split
  
  # 从指定位置开始分割
  python split_fvecs.py -i data/large.fvecs -o data/splits -s 1000000 --start 2000000
  
  # 根据内存限制自动建议分割大小
  python split_fvecs.py -i data/large.fvecs -o data/splits --suggest-size --memory 8
  
  # 仅显示文件信息，不进行分割
  python split_fvecs.py -i data/large.fvecs --info-only
        """,
    )
    
    parser.add_argument("-i", "--input", required=True, 
                       help="输入的 .fvecs 文件路径")
    parser.add_argument("-o", "--output", 
                       help="输出目录路径")
    parser.add_argument("-s", "--size", type=int,
                       help="每个分割文件的向量数量")
    parser.add_argument("-p", "--prefix", 
                       help="输出文件名前缀（默认使用输入文件名）")
    parser.add_argument("--start", type=int, default=0,
                       help="起始向量索引（默认: 0）")
    parser.add_argument("--suggest-size", action="store_true",
                       help="根据内存限制建议分割大小")
    parser.add_argument("--memory", type=float, default=4.0,
                       help="内存限制 (GB，用于建议分割大小，默认: 4.0)")
    parser.add_argument("--info-only", action="store_true",
                       help="仅显示文件信息，不进行分割")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    if not args.input.endswith('.fvecs'):
        print(f"❌ 错误: 输入文件必须是 .fvecs 格式")
        return
    
    # 获取文件信息
    try:
        total_vectors, dimension = get_fvecs_info(args.input)
    except Exception as e:
        print(f"❌ 错误: 无法读取文件信息: {e}")
        return
    
    print(f"📁 文件信息:")
    print(f"  文件路径: {args.input}")
    print(f"  总向量数: {total_vectors:,}")
    print(f"  向量维度: {dimension}")
    print(f"  文件大小: {os.path.getsize(args.input) / (1024**3):.2f} GB")
    
    # 如果只显示信息，直接返回
    if args.info_only:
        return
    
    # 建议分割大小
    if args.suggest_size:
        suggested_size = suggest_chunk_size(total_vectors, dimension, args.memory)
        memory_info = estimate_memory_usage(total_vectors, dimension, suggested_size)
        
        print(f"\n💡 内存分析 (限制: {args.memory} GB):")
        print(f"  建议分割大小: {suggested_size:,} 向量")
        print(f"  单块内存需求: {memory_info['chunk_memory_gb']:.2f} GB")
        print(f"  总内存需求: {memory_info['total_memory_gb']:.2f} GB")
        print(f"  预计分割块数: {math.ceil(total_vectors / suggested_size)}")
        
        if not args.size:
            response = input(f"\n是否使用建议的分割大小 {suggested_size:,}? (y/n): ")
            if response.lower() in ['y', 'yes']:
                args.size = suggested_size
            else:
                print("已取消分割操作")
                return
    
    # 检查必需参数
    if not args.size:
        print(f"❌ 错误: 请指定分割大小 (-s) 或使用 --suggest-size")
        return
    
    if not args.output:
        print(f"❌ 错误: 请指定输出目录 (-o)")
        return
    
    # 验证分割大小
    if args.size <= 0:
        print(f"❌ 错误: 分割大小必须大于 0")
        return
    
    if args.size > total_vectors:
        print(f"⚠️ 警告: 分割大小 ({args.size:,}) 大于总向量数 ({total_vectors:,})")
        print(f"将使用总向量数作为分割大小")
        args.size = total_vectors
    
    # 显示内存估算
    memory_info = estimate_memory_usage(total_vectors, dimension, args.size)
    print(f"\n💾 内存估算:")
    print(f"  单块内存需求: {memory_info['chunk_memory_gb']:.2f} GB")
    print(f"  总内存需求: {memory_info['total_memory_gb']:.2f} GB")
    
    if memory_info['total_memory_gb'] > args.memory:
        print(f"  ⚠️ 警告: 估算内存需求超过限制 ({args.memory} GB)")
        response = input("是否继续? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("已取消分割操作")
            return
    
    # 执行分割
    try:
        output_files = split_fvecs_file(
            args.input, 
            args.output, 
            args.size, 
            args.prefix, 
            args.start
        )
        
        print(f"\n🎉 分割成功完成!")
        print(f"生成了 {len(output_files)} 个文件")
        
    except Exception as e:
        print(f"❌ 分割过程中出现错误: {e}")
        return


if __name__ == "__main__":
    main()