#!/usr/bin/env python3
import argparse
import os
import sys
import math
import numpy as np
from faiss_benchmark.utils import (
    get_fvecs_info, fvecs_read_range, fvecs_write_streaming
)


def merge_fvecs(input_a: str, input_b: str, output_path: str, chunk_size: int = 100000, overwrite: bool = False):
    # Validate inputs
    if not os.path.exists(input_a):
        raise FileNotFoundError(f"input_a not found: {input_a}")
    if not os.path.exists(input_b):
        raise FileNotFoundError(f"input_b not found: {input_b}")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"output file exists: {output_path}. Use --overwrite to replace.")

    # Inspect both files
    num_a, dim_a = get_fvecs_info(input_a)
    num_b, dim_b = get_fvecs_info(input_b)
    if dim_a != dim_b:
        raise RuntimeError(f"维度不一致: A={dim_a}, B={dim_b}")
    total = num_a + num_b
    print(f"A: {os.path.basename(input_a)} -> {num_a:,} x {dim_a}")
    print(f"B: {os.path.basename(input_b)} -> {num_b:,} x {dim_b}")
    print(f"输出: {output_path}")

    # Generator that yields batches from A then B
    def vectors_generator():
        processed = 0
        # Read A
        a_offset = 0
        while a_offset < num_a:
            bs = min(chunk_size, num_a - a_offset)
            batch = fvecs_read_range(input_a, a_offset, bs)
            yield batch
            a_offset += bs
            processed += batch.shape[0]
            if processed % (chunk_size * 10) == 0 or a_offset >= num_a:
                print(f"  A 进度: {a_offset:,}/{num_a:,}")

        # Read B
        b_offset = 0
        while b_offset < num_b:
            bs = min(chunk_size, num_b - b_offset)
            batch = fvecs_read_range(input_b, b_offset, bs)
            yield batch
            b_offset += bs
            processed += batch.shape[0]
            if processed % (chunk_size * 10) == 0 or b_offset >= num_b:
                print(f"  B 进度: {b_offset:,}/{num_b:,}")

    # Stream write
    print(f"开始合并，chunk_size={chunk_size:,}，总计 {total:,} 向量")
    fvecs_write_streaming(output_path, vectors_generator(), total_count=total)
    print("🎉 合并完成!")


def main():
    parser = argparse.ArgumentParser(
        description="合并两个 .fvecs 文件为一个（流式处理，避免内存溢出）"
    )
    parser.add_argument("--input-a", required=True, help="输入文件 A 的路径 (.fvecs)")
    parser.add_argument("--input-b", required=True, help="输入文件 B 的路径 (.fvecs)")
    parser.add_argument("--output", required=True, help="输出合并后的 .fvecs 文件路径")
    parser.add_argument("--chunk-size", type=int, default=100000, help="读取与写入的批大小（默认 100000）")
    parser.add_argument("--overwrite", action="store_true", help="如果输出文件已存在则覆盖")

    args = parser.parse_args()
    try:
        merge_fvecs(args.input_a, args.input_b, args.output, chunk_size=args.chunk_size, overwrite=args.overwrite)
    except Exception as e:
        print(f"合并失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

