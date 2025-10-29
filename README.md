# Faiss Benchmark

一个轻量级、模块化且易于使用的 Faiss 基准测试框架，支持真实数据集和配置化测试。

## 安装

1. **克隆仓库：**
   ```bash
   git clone <repository_url>
   cd faiss-benchmark
   ```

2. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

## 数据集准备

### 数据集格式

数据集应放置在 `data/` 目录下，每个数据集需要包含三个文件：

- `{dataset_name}_base.fvecs` - 检索集（基础向量）
- `{dataset_name}_query.fvecs` - 查询集
- `{dataset_name}_groundtruth.ivecs` - 真实答案集

### 示例：SIFT 数据集

```
data/
├── sift_base.fvecs
├── sift_query.fvecs
└── sift_groundtruth.ivecs
```

您可以从以下地址下载常用数据集：
- [SIFT1M](http://corpus-texmex.irisa.fr/)
- [Deep1B](http://sites.skoltech.ru/compvision/noimi/)

## 配置文件

通过 `config.yaml` 文件配置要测试的数据集、算法、搜索参数和 CPU 线程数：

```yaml
dataset: "sift"
topk: 10
num_threads: 4

index_types:
  - "Flat"
  - "IVF1024,Flat"
  - "HNSW32,Flat"
```

- `dataset`: 要加载的数据集名称（不含扩展名）。程序会自动在 `data/` 目录下查找 `{dataset_name}_base.fvecs`、`{dataset_name}_query.fvecs` 和 `{dataset_name}_groundtruth.ivecs`。
- `topk`: 指定在搜索时返回的最近邻结果数量。这个值会影响召回率的计算（例如 `Recall@10`）。
- `num_threads`: 指定 Faiss 在 CPU 上进行计算时可以使用的线程数。这对于在多核 CPU 上加速索引构建和搜索非常有用。
- `index_types`: 一个包含要测试的 Faiss 索引字符串的列表。

### 支持的索引类型

- `Flat`: 精确搜索
- `PCAR64,Flat`: PCA降维到64维后精确搜索
- `IVF1024,Flat`: 1024个聚类中心的倒排索引，精确搜索
- `IVF1024,PQ8+16`: 倒排索引 + 乘积量化
- `HNSW32,Flat`: 32邻居的分层导航小世界图，精确搜索
- `HNSW32,PQ8+16`: HNSW + 乘积量化

## 使用方法

### 基本使用

```bash
python main.py
```

### 使用 GPU 加速

```bash
python main.py --gpu
```

### 指定配置文件

```bash
python main.py --config my_config.yaml
```

### 完整示例

```bash
python main.py --config config.yaml --gpu
```

## 输出结果

基准测试将输出以下指标：

- **Training time**: 索引训练时间
- **Adding time**: 向量添加时间
- **Search time**: 搜索时间
- **QPS**: 每秒查询数
- **Recall@10**: 前10个结果的召回率

## 自定义

### 添加新的数据集

1. 将数据集文件放入 `data/` 目录
2. 在 `config.yaml` 中修改 `dataset` 字段

### 添加新的索引类型

在 `config.yaml` 的 `index_types` 列表中添加新的 Faiss 索引字符串。

### 修改搜索参数

可以在 `benchmark.py` 中的 `run_benchmark` 函数中修改 `k` 参数来改变返回的结果数量。

## 工具

### generate_dataset.py

数据集生成工具，用于将单个 `.fvecs` 文件分割成 query 和 base 两部分，并生成对应的 groundtruth 文件。支持 GPU 加速和分批计算，适用于大规模数据集。

#### 功能
- 将 `.fvecs` 文件按指定数量分割为 query 和 base 部分
- 使用 Faiss 暴力搜索生成准确的 groundtruth
- **GPU 加速**: 支持使用 GPU 显著加速 groundtruth 生成过程
- **分批计算**: 支持大规模数据集的分批处理，避免内存溢出
- **内存估算**: 自动估算内存使用量并建议合适的批处理大小
- **进度显示**: 实时显示处理进度
- 支持自定义 topk 值
- 自动验证输入文件格式和数据完整性

#### 使用方法

```bash
# CPU 模式
python generate_dataset.py -i <input_file> -q <num_queries> -k <topk> -o <output_prefix>

# GPU 模式
python generate_dataset.py -i <input_file> -q <num_queries> -k <topk> -o <output_prefix> --gpu

# 自定义批处理大小
python generate_dataset.py -i <input_file> -q <num_queries> -k <topk> -o <output_prefix> --gpu --batch-size 500
```

#### 参数说明
- `-i, --input`: 输入的 .fvecs 文件路径
- `-q, --queries`: query 集合的向量数量
- `-k, --topk`: 每个 query 的最近邻数量 (默认: 100)
- `-o, --output`: 输出文件前缀 (不含扩展名)
- `--gpu`: 使用 GPU 加速 groundtruth 生成
- `--batch-size`: 批处理大小 (0 表示自动选择，默认: 0)
- `--memory-limit`: 内存限制 (GB，用于自动选择批处理大小，默认: 8.0)

#### 示例

```bash
# CPU 模式：从 sift.fvecs 中提取 1000 个 query，生成 top-100 groundtruth
python generate_dataset.py -i data/sift.fvecs -q 1000 -k 100 -o data/sift

# GPU 模式：使用 GPU 加速，自动选择批处理大小
python generate_dataset.py -i data/sift.fvecs -q 1000 -k 100 -o data/sift --gpu

# GPU 模式：自定义批处理大小为 500
python generate_dataset.py -i data/sift.fvecs -q 1000 -k 100 -o data/sift --gpu --batch-size 500

# 大数据集：限制内存使用为 16GB
python generate_dataset.py -i data/large_dataset.fvecs -q 10000 -k 100 -o data/large --gpu --memory-limit 16.0
```

这将生成以下文件：
- `data/sift_query.fvecs`: 包含 1000 个 query 向量
- `data/sift_base.fvecs`: 包含剩余的 base 向量
- `data/sift_groundtruth.ivecs`: 每个 query 的前 100 个最近邻索引

#### 性能优化建议

1. **GPU 加速**: 对于大规模数据集，强烈建议使用 `--gpu` 参数，可以显著提升处理速度
2. **批处理大小**: 
   - 自动模式 (`--batch-size 0`) 会根据可用内存自动选择合适的批处理大小
   - 手动设置时，建议根据 GPU 内存大小调整：
     - 8GB GPU: 500-1000
     - 16GB GPU: 1000-2000
     - 24GB+ GPU: 2000+
3. **内存管理**: 使用 `--memory-limit` 参数限制内存使用，避免系统内存不足

#### 技术特点

- **准确性**: 使用 Faiss 的 `IndexFlatL2` 进行精确的 L2 距离计算
- **可扩展性**: 支持处理任意大小的数据集，通过分批计算避免内存限制
- **兼容性**: 生成的文件格式与标准基准测试数据集完全兼容
- **错误处理**: 完善的输入验证和错误提示

### 数据集切割工具

项目提供了一个 `split_dataset.py` 工具，用于将 `.fvecs` 和 `.ivecs` 文件按指定的条目数进行切割，方便创建小规模的测试样本。

#### 使用方法

```bash
python split_dataset.py --input <input_file> --output <output_file> --count <num_entries>
```

- `--input`: 输入文件路径（例如 `data/sift_base.fvecs`）
- `--output`: 输出文件路径（例如 `data/sift_base_10k.fvecs`）
- `--count`: 要保留的条目数量

#### 示例

将 `sift_base.fvecs` 切割为包含 10000 个向量的新文件：

```bash
python split_dataset.py -i data/sift_base.fvecs -o data/sift_base_10k.fvecs -c 10000
```

## 项目结构

```
faiss-benchmark/
├── README.md
├── config.yaml          # 配置文件
├── main.py              # 主入口
├── requirements.txt     # 依赖列表
├── data/               # 数据集目录
└── faiss_benchmark/    # 核心模块
    ├── __init__.py
    ├── datasets.py     # 数据集加载
    ├── indexes.py      # 索引创建
    ├── benchmark.py    # 基准测试
    └── results.py      # 结果输出
```