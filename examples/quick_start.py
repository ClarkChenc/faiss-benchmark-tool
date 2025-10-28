#!/usr/bin/env python3
"""
FAISS Benchmark Framework - 快速开始示例

这个脚本演示了如何快速开始使用 FAISS 基准测试框架。
它创建一个简单的数据集，运行基本的基准测试，并生成结果报告。
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from faiss_benchmark import BenchmarkRunner
from faiss_benchmark.utils.logger import get_logger
from faiss_benchmark.utils.io_utils import write_fvecs, write_ivecs


def create_sample_dataset(output_dir="datasets", dataset_name="quick_start"):
    """创建一个简单的示例数据集"""
    print("🔧 创建示例数据集...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据集参数
    dimension = 128
    nb_base = 10000      # 基础向量数量
    nb_query = 100       # 查询向量数量
    k = 10               # 每个查询的真实近邻数量
    
    # 生成随机数据
    np.random.seed(42)
    base_vectors = np.random.random((nb_base, dimension)).astype('float32')
    query_vectors = np.random.random((nb_query, dimension)).astype('float32')
    
    # 计算真实的最近邻（使用暴力搜索）
    print("  计算真实最近邻...")
    distances = np.linalg.norm(
        query_vectors[:, np.newaxis, :] - base_vectors[np.newaxis, :, :], 
        axis=2
    )
    ground_truth = np.argsort(distances, axis=1)[:, :k].astype('int32')
    
    # 保存数据集文件
    base_file = os.path.join(output_dir, f"{dataset_name}_base.fvecs")
    query_file = os.path.join(output_dir, f"{dataset_name}_query.fvecs")
    gt_file = os.path.join(output_dir, f"{dataset_name}_groundtruth.ivecs")
    
    write_fvecs(base_file, base_vectors)
    write_fvecs(query_file, query_vectors)
    write_ivecs(gt_file, ground_truth)
    
    print(f"  ✅ 数据集已创建:")
    print(f"    基础向量: {base_file} ({nb_base} vectors, {dimension}D)")
    print(f"    查询向量: {query_file} ({nb_query} vectors, {dimension}D)")
    print(f"    真实结果: {gt_file} (top-{k} for each query)")
    
    return {
        "base_file": base_file,
        "query_file": query_file,
        "ground_truth_file": gt_file,
        "dimension": dimension,
        "nb_base": nb_base,
        "nb_query": nb_query
    }


def run_quick_benchmark():
    """运行快速基准测试"""
    print("\n🚀 开始快速基准测试...")
    
    # 创建示例数据集
    dataset_info = create_sample_dataset()
    
    # 初始化基准测试运行器
    print("\n📊 初始化基准测试运行器...")
    runner = BenchmarkRunner("config.yaml")
    
    # 添加数据集
    dataset_name = "quick_start"
    runner.dataset_manager.add_dataset(
        name=dataset_name,
        base_file=dataset_info["base_file"],
        query_file=dataset_info["query_file"],
        ground_truth_file=dataset_info["ground_truth_file"]
    )
    
    # 定义要测试的索引配置
    index_configs = [
        {"type": "Flat", "params": {}},
        {"type": "IVFFlat", "params": {"nlist": 100, "nprobe": 10}},
        {"type": "IVFPQ", "params": {"nlist": 100, "m": 8, "nbits": 8, "nprobe": 10}},
    ]
    
    # 运行基准测试
    results = []
    for i, index_config in enumerate(index_configs, 1):
        print(f"\n🔍 运行测试 {i}/{len(index_configs)}: {index_config['type']}")
        
        try:
            # CPU 测试
            result_cpu = runner.run_single_benchmark(
                dataset_name=dataset_name,
                index_config=index_config,
                hardware_type="cpu",
                k=10
            )
            result_cpu['hardware_type'] = 'cpu'
            results.append(result_cpu)
            
            print(f"  CPU - QPS: {result_cpu['qps']:.1f}, "
                  f"Recall: {result_cpu['recall']:.3f}, "
                  f"构建时间: {result_cpu['index_build_time']:.2f}s")
            
            # GPU 测试（如果可用）
            try:
                result_gpu = runner.run_single_benchmark(
                    dataset_name=dataset_name,
                    index_config=index_config,
                    hardware_type="gpu",
                    k=10
                )
                result_gpu['hardware_type'] = 'gpu'
                results.append(result_gpu)
                
                print(f"  GPU - QPS: {result_gpu['qps']:.1f}, "
                      f"Recall: {result_gpu['recall']:.3f}, "
                      f"构建时间: {result_gpu['index_build_time']:.2f}s")
                      
            except Exception as e:
                print(f"  GPU 测试跳过: {e}")
                
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
    
    return results


def generate_quick_report(results):
    """生成快速报告"""
    print("\n📋 生成测试报告...")
    
    if not results:
        print("  ❌ 没有可用的测试结果")
        return
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 按性能指标排序
    print("\n🏆 性能排行榜:")
    print("-" * 60)
    print(f"{'索引类型':<15} {'硬件':<6} {'QPS':<8} {'Recall':<8} {'构建时间':<8}")
    print("-" * 60)
    
    # 按 QPS 排序
    sorted_results = sorted(results, key=lambda x: x['qps'], reverse=True)
    for result in sorted_results:
        print(f"{result['index_name']:<15} "
              f"{result['hardware_type']:<6} "
              f"{result['qps']:<8.1f} "
              f"{result['recall']:<8.3f} "
              f"{result['index_build_time']:<8.2f}")
    
    # 找出最佳性能者
    best_qps = max(results, key=lambda x: x['qps'])
    best_recall = max(results, key=lambda x: x['recall'])
    fastest_build = min(results, key=lambda x: x['index_build_time'])
    
    print(f"\n🎯 最佳性能:")
    print(f"  最高 QPS: {best_qps['index_name']} ({best_qps['hardware_type']}) - {best_qps['qps']:.1f}")
    print(f"  最高 Recall: {best_recall['index_name']} ({best_recall['hardware_type']}) - {best_recall['recall']:.3f}")
    print(f"  最快构建: {fastest_build['index_name']} ({fastest_build['hardware_type']}) - {fastest_build['index_build_time']:.2f}s")
    
    # 保存详细结果
    import json
    results_file = "results/quick_start_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {results_file}")


def main():
    """主函数"""
    print("🎉 欢迎使用 FAISS Benchmark Framework!")
    print("这是一个快速开始示例，将演示基本的基准测试功能。")
    
    try:
        # 运行快速基准测试
        results = run_quick_benchmark()
        
        # 生成报告
        generate_quick_report(results)
        
        print("\n✅ 快速开始示例完成!")
        print("\n📚 接下来你可以:")
        print("  1. 查看 examples/basic_benchmark.py 了解更多基础功能")
        print("  2. 查看 examples/advanced_benchmark.py 了解高级功能")
        print("  3. 修改 config.yaml 来自定义配置")
        print("  4. 使用 CLI 工具: python -m faiss_benchmark.cli --help")
        
    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()