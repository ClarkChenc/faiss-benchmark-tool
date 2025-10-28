# FAISS Benchmark Framework

一个基于 FAISS 库的综合性能基准测试框架，用于比较不同数据集和算法下 GPU 和 CPU 的检索性能。

## 特性

- 🚀 **多算法支持**: 支持 Flat、IVF、HNSW、LSH 等多种 FAISS 索引算法
- 📊 **性能指标**: 全面的性能指标包括 QPS、Recall、构建时间、内存使用等
- 🔄 **硬件对比**: 支持 CPU 和 GPU 性能对比测试
- 📈 **可视化分析**: 丰富的静态和交互式图表，包括帕累托前沿分析
- 📁 **标准格式**: 统一使用 fvecs/ivecs 格式的数据集
- 🎯 **横向比较**: 便于不同算法间的性能横向比较

## 安装

### 依赖要求

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `faiss-cpu` / `faiss-gpu`: FAISS 库
- `numpy`, `pandas`: 数据处理
- `matplotlib`, `seaborn`, `plotly`: 可视化
- `scikit-learn`: 机器学习工具
- `pyyaml`: 配置文件解析

### GPU 支持

如需 GPU 支持，请安装 `faiss-gpu`：

```bash
pip install faiss-gpu
```

## 快速开始

### 基础使用

```python
from faiss_benchmark import BenchmarkRunner

# 初始化基准测试运行器
runner = BenchmarkRunner("config.yaml")

# 添加数据集
runner.dataset_manager.add_dataset(
    name="my_dataset",
    base_file="data/my_dataset_base.fvecs",
    query_file="data/my_dataset_query.fvecs",
    ground_truth_file="data/my_dataset_groundtruth.ivecs"
)

# 运行单个基准测试
result = runner.run_single_benchmark(
    dataset_name="my_dataset",
    index_config={"type": "IVFFlat", "params": {"nlist": 100}},
    hardware_type="cpu",
    k=10
)

print(f"QPS: {result['qps']:.2f}")
print(f"Recall: {result['recall']:.3f}")
```

### 运行示例

```bash
# 基础示例
python examples/basic_benchmark.py

# 高级示例（包含多数据集、参数扫描等）
python examples/advanced_benchmark.py
```

## 数据集格式

框架使用标准的 fvecs/ivecs 格式：

- **base.fvecs**: 基础向量集
- **query.fvecs**: 查询向量集  
- **groundtruth.ivecs**: 真实最近邻结果

### 格式说明

- **fvecs**: 32位浮点向量格式
- **ivecs**: 32位整数向量格式

每个文件的格式为：`[dimension][vector1][dimension][vector2]...`

## 配置文件

`config.yaml` 示例：

```yaml
# 数据集配置
datasets:
  sift1m:
    base_file: "datasets/sift1m_base.fvecs"
    query_file: "datasets/sift1m_query.fvecs"
    ground_truth_file: "datasets/sift1m_groundtruth.ivecs"

# 索引配置
indexes:
  - type: "Flat"
    params: {}
  
  - type: "IVFFlat"
    params:
      nlist: 100
      nprobe: 10
  
  - type: "IVFPQ"
    params:
      nlist: 100
      m: 8
      nbits: 8
      nprobe: 10

# 基准测试参数
benchmark:
  k_values: [1, 10, 100]
  hardware_types: ["cpu", "gpu"]
  warmup_queries: 100
  test_queries: 1000

# 硬件配置
hardware:
  cpu_threads: 4
  gpu_device: 0
```

## 支持的索引类型

| 索引类型 | 描述 | 主要参数 |
|---------|------|---------|
| Flat | 暴力搜索 | 无 |
| IVFFlat | 倒排文件索引 | nlist, nprobe |
| IVFPQ | 乘积量化倒排索引 | nlist, m, nbits, nprobe |
| HNSW | 分层导航小世界图 | M, efConstruction, efSearch |
| LSH | 局部敏感哈希 | nbits |

## 性能指标

框架提供以下性能指标：

- **QPS**: 每秒查询数
- **Recall@k**: 召回率
- **Precision@k**: 精确率  
- **Search Time**: 平均搜索时间
- **Index Build Time**: 索引构建时间
- **Memory Usage**: 内存使用量
- **GPU Memory**: GPU 内存使用（如适用）

## 可视化功能

### 静态图表

```python
from faiss_benchmark.visualization import BenchmarkPlotter

plotter = BenchmarkPlotter(results_manager)

# QPS 比较
plotter.plot_performance_comparison(metric='qps', group_by='index_name')

# 散点图分析
plotter.plot_scatter_analysis(x_metric='search_time', y_metric='recall')

# 帕累托前沿
plotter.plot_pareto_frontier(x_metric='search_time', y_metric='recall')
```

### 交互式图表

```python
# 交互式比较图
plotter.plot_performance_comparison(
    metric='qps', 
    group_by='index_name',
    interactive=True
)

# 综合仪表板
plotter.create_dashboard()
```

## 结果分析

### 算法比较

```python
from faiss_benchmark.visualization import ResultsAnalyzer

analyzer = ResultsAnalyzer(results_manager)

# 性能趋势分析
trends = analyzer.analyze_performance_trends('qps', 'index_name')

# 算法比较
comparison = analyzer.compare_algorithms(
    dataset_name='sift1m',
    metrics=['qps', 'recall', 'search_time']
)

# 帕累托最优解
pareto_optimal = analyzer.find_pareto_optimal('search_time', 'recall')
```

### 硬件效率分析

```python
# CPU vs GPU 效率分析
hw_analysis = analyzer.analyze_hardware_efficiency()

# 可扩展性分析
scalability = analyzer.analyze_scalability('dataset_size', 'qps')
```

## 项目结构

```
faiss-benchmark/
├── faiss_benchmark/           # 主要框架代码
│   ├── datasets/             # 数据集管理
│   ├── indexes/              # 索引管理
│   ├── benchmarks/           # 基准测试
│   ├── utils/                # 工具函数
│   └── visualization/        # 可视化模块
├── examples/                 # 示例脚本
├── datasets/                 # 数据集存储
├── results/                  # 结果输出
├── plots/                    # 图表输出
├── config.yaml              # 配置文件
└── requirements.txt          # 依赖列表
```

## 高级功能

### 参数扫描

```python
# 自动参数扫描
param_combinations = index_manager.generate_param_combinations('IVFFlat')
for params in param_combinations:
    result = runner.run_single_benchmark(dataset, index_config, 'cpu')
```

### 批量测试

```python
# 批量测试多个数据集
results = runner.run_full_benchmark(['sift1m', 'glove', 'random'])
```

### 结果过滤和分析

```python
# 过滤结果
filtered_results = results_manager.filter_results(
    dataset_name='sift1m',
    hardware_type='gpu'
)

# 获取最佳性能者
best_performers = analyzer.get_best_performers(
    metric='qps',
    group_by='dataset_name'
)
```

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 常见问题

### Q: 如何添加新的索引类型？

A: 在 `faiss_benchmark/indexes/index_manager.py` 中的 `_create_index` 方法中添加新的索引类型支持。

### Q: 如何自定义性能指标？

A: 在 `faiss_benchmark/benchmarks/metrics.py` 中的 `MetricsCalculator` 类中添加新的指标计算方法。

### Q: GPU 测试失败怎么办？

A: 确保已安装 `faiss-gpu` 并且系统有可用的 CUDA GPU。可以通过 `nvidia-smi` 检查 GPU 状态。

### Q: 如何处理大型数据集？

A: 框架支持数据缓存和分批处理。可以在配置文件中调整 `test_queries` 参数来限制测试查询数量。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue: [GitHub Issues](https://github.com/your-repo/faiss-benchmark/issues)
- 邮件: your-email@example.com

---

**注意**: 本框架仅用于研究和基准测试目的。在生产环境中使用前请充分测试。