# Faiss Benchmark

一个轻量级、模块化且易于使用的 Faiss 基准测试框架，支持真实数据集和配置化测试。

## 安装

1. **克隆仓库：**
   ```bash
   git clone <repository_url>
   cd faiss-benchmark
   ```

2. **安装依赖：**
   
   faiss-gpu: (支持 cuda11.4 及以上)
   conda install -c conda-forge faiss-gpu
   
   faiss-gpu-cuvs: （需要保证支持 cuda12.0 及以上）
   conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.12.0

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

### 配置文件管理

为了避免 git 管理冲突，本项目采用模板文件方式管理配置：

- **`config.yaml.template`**: 配置模板文件（受 git 管理）
- **`config.yaml`**: 实际配置文件（被 git 忽略，用户自定义）

#### 首次使用

当您首次运行程序时，如果 `config.yaml` 不存在，程序会自动从 `config.yaml.template` 创建：

```bash
python main.py --config config.yaml
# 输出：配置文件 config.yaml 不存在，正在从模板 config.yaml.template 创建...
# 输出：已创建配置文件 config.yaml，请根据需要修改参数。
```

#### 手动创建配置文件

您也可以手动复制模板文件：

```bash
cp config.yaml.template config.yaml
```

然后根据需要修改 `config.yaml` 中的参数。

#### 配置文件结构

通过 `config.yaml` 文件配置要测试的数据集、算法、搜索参数和 CPU 线程数：

- **`dataset`**: 数据集名称，对应 `data/` 目录下的数据集文件前缀。
- **`topk`**: 搜索时返回的最近邻居数量。
- **`latency_batch_size`**: 延迟统计的微批大小（用于计算 avg/p99 延迟），默认 32。
- **`warmup_queries`**: 预热查询数量（在正式计时前先跑若干查询，以稳定 QPS/延迟），默认 0（关闭）。
- **`num_threads`**: Faiss 使用的 CPU 线程数。
- **`batch_processing`**: 批处理配置，用于内存优化：
  - **`enabled`**: 是否启用批处理模式 (`true` / `false`)。
  - **`batch_size`**: 每批处理的向量数量。
  - **`train_batch_size`**: 训练时使用的批大小。
- **`index_types`**: 一个包含多个索引配置的列表，每个配置包含：
  - **`index_type`**: Faiss 索引的类型，例如 `"Flat"`, `"IVF1024,Flat"`, `"HNSW32,Flat"`。
  - **`use_gpu`**: 是否为该索引使用 GPU（可选，优先于 `index_param` 中的同名字段）。
  - **`index_param`**: 构建阶段参数，只影响索引结构；参与缓存键（例如 HNSW 的 `efConstruction`、IVF 的 `nlist`、HNSWLIB 的 `M`/`efConstruction`/`space`）。
  - **`search_param`**: 搜索阶段参数，仅影响查询行为；不参与缓存键（例如 HNSW 的 `efSearch`、IVF 的 `nprobe`、通用的 `latency_batch_size`）。
  - 兼容说明：仍然支持旧的 **`params`** 字段，程序会自动拆分为上述两类参数。

### 索引参数分类

参数分为两类，分别在构建阶段与搜索阶段生效（另有全局统计配置）：

- 构建参数（`index_param`）：影响索引结构与缓存键
  - HNSW：`efConstruction`
  - HNSWLIB：`M`、`efConstruction`、`space`（`l2` 或 `cosine`）
  - IVF：`nlist`（如需覆盖；通常由 `index_type` 指定）
- 搜索参数（`search_param`）：影响查询行为，不参与缓存键
  - HNSW：`efSearch`
  - HNSWLIB：`efSearch`
  - IVF：`nprobe`
  - （无）

全局统计配置：
- `latency_batch_size`：用于延迟统计的微批大小（越大近似越粗但吞吐更稳定），默认 32

示例配置：
```yaml
dataset: "sift"
topk: 10
latency_batch_size: 32
warmup_queries: 0
num_threads: 4

index_types:
  - index_type: "Flat"
    index_param: {}
    search_param: {}

  - index_type: "IVF1024,Flat"
    use_gpu: true
    index_param: {}
    search_param:
      nprobe: 16

  - index_type: "HNSW32,Flat"
    index_param:
      efConstruction: 40
    search_param:
      efSearch: 16

  - index_type: "HNSWLIB"
    index_param:
      M: 16
      efConstruction: 200
      space: l2
    search_param:
      efSearch: 32
```

- `dataset`: 要加载的数据集名称（不含扩展名）。程序会自动在 `data/` 目录下查找 `{dataset_name}_base.fvecs`、`{dataset_name}_query.fvecs` 和 `{dataset_name}_groundtruth.ivecs`。
- `topk`: 指定在搜索时返回的最近邻结果数量。这个值会影响召回率的计算（例如 `Recall@10`）。
- `num_threads`: 指定 Faiss 在 CPU 上进行计算时可以使用的线程数。这对于在多核 CPU 上加速索引构建和搜索非常有用。
- `index_types`: 一个包含要测试的 Faiss 索引配置的列表，支持每个索引分别设置 GPU 与参数。

### 支持的索引类型

- `Flat`: 精确搜索
- `PCAR64,Flat`: PCA降维到64维后精确搜索
- `IVF1024,Flat`: 1024个聚类中心的倒排索引，精确搜索
- `IVF1024,PQ8+16`: 倒排索引 + 乘积量化
- `HNSW32,Flat`: 32邻居的分层导航小世界图，精确搜索
- `HNSW32,PQ8+16`: HNSW + 乘积量化
 - `CAGRA`: GPU 图索引（cuVS/RAFT）；或 `CAGRA->HNSW32,Flat` 转换为 CPU HNSW 检索与缓存
 - `SCANN`: Google ScaNN（CPU ANN，支持 dot_product 与 squared_l2）

## 使用方法

### 基本使用

```bash
python main.py
```

### 使用 GPU 加速

在 `config.yaml` 中为指定的索引条目设置 `use_gpu: true`：

```yaml
index_types:
  - index_type: "Flat"
    use_gpu: true
    index_param: {}
    search_param: {}

### 使用 ScaNN (CPU)

在 `config.yaml` 增加如下条目即可使用 ScaNN：

```yaml
index_types:
  - index_type: "SCANN"
    use_gpu: false
    index_param:
      distance_measure: dot_product   # 或 squared_l2
      num_neighbors: 10               # 构建候选邻居数
      num_leaves: 2000                # 可选：分区数量
      leaf_size: 100                  # 可选：叶子大小
      ah_bits: 2                      # 可选：AH 位数（1-4 常见）
      ah_threshold: 0.2               # 可选：AH 阈值
      reorder_k: 100                  # 可选：构建时开启重排，候选规模
    search_param:
      final_num_neighbors: 10         # 搜索阶段最终返回邻居数量（默认跟 topk）
```

依赖：在环境中安装 `scann` 包（Linux 轮子，Python 3.8+）。
```

### 指定配置文件

```bash
python main.py --config my_config.yaml
```

### 完整示例

```bash
python main.py --config config.yaml --gpu
```

## 批处理模式（内存优化）

对于大型数据集，传统的全量加载方式可能会导致内存不足（OOM）问题。本工具提供了批处理模式来解决这个问题。

### 启用批处理模式

在 `config.yaml` 中设置批处理配置：

```yaml
batch_processing:
  enabled: true           # 启用批处理模式
  batch_size: 100000     # 每批处理 10万个向量
  train_batch_size: 500000 # 训练时使用 50万个向量
```

### 批处理模式的优势

1. **内存优化**: 避免一次性加载整个数据集到内存，显著降低内存使用量
2. **避免 OOM**: 特别适用于处理大型数据集（如 Deep1B）
3. **灵活配置**: 可根据可用内存调整批次大小

### 批处理参数说明

- **`batch_size`**: 控制每次添加到索引的向量数量
  - 较小值（如 50K）：内存使用更少，但处理时间可能稍长
  - 较大值（如 500K）：处理更快，但需要更多内存
  - 建议根据可用内存设置，通常 50K-500K 之间

- **`train_batch_size`**: 控制训练索引时使用的向量数量
  - 通常可以设置得比 `batch_size` 更大
  - 训练阶段相比添加阶段通常需要更少的内存

### 使用建议

- 对于小型数据集（如 SIFT1M），可以保持 `enabled: false` 使用传统模式
- 对于大型数据集（如 Deep1B），建议启用批处理模式
- 如果遇到内存不足错误，可以减小 `batch_size` 和 `train_batch_size`

## 索引缓存

为了提高效率，该工具实现了一个索引缓存机制：当您第一次使用特定的数据集、索引类型和“构建参数”组合运行基准测试时，构建好的索引将被自动保存到 `index_cache` 目录中。

在后续运行中，如果检测到匹配的缓存文件，程序将直接加载该索引，从而跳过耗时的训练与向量添加过程。

缓存文件名模式：

- `dataset_indexType_buildParamStr.index`（CPU 模式）
- `dataset_indexType_buildParamStr_gpu.index`（GPU 标识会出现在文件名中，但默认不缓存 GPU 模式，见下）

缓存策略：
- 仅“构建参数”（`index_param`）与 GPU 标志参与缓存键；“搜索参数”（`search_param`）不改变缓存键，可复用索引。
- GPU 模式索引不缓存（避免跨设备兼容问题），每次都重新构建。
- 若要强制重建索引，删除 `index_cache/` 目录下对应文件即可。

## 输出结果

基准测试将输出以下指标（延迟采用微批近似，由全局 `latency_batch_size` 控制）：

- **Training time**: 索引训练时间
- **Adding time**: 向量添加时间
- **Search time**: 搜索时间
- **QPS**: 每秒查询数
- **Latency avg (ms)**: 单查询平均延迟（毫秒）
- **Latency p99 (ms)**: 单查询 p99 延迟（毫秒）
- **Recall@10**: 前10个结果的召回率

## 自定义

### 添加新的数据集

1. 将数据集文件放入 `data/` 目录
2. 在 `config.yaml` 中修改 `dataset` 字段

### 添加新的索引类型

在 `config.yaml` 的 `index_types` 列表中添加新的 Faiss 索引字符串。

#### CAGRA（GPU，仅 cuVS；需要 CUDA 12）

- 本项目支持通过适配器集成 NVIDIA cuVS 的 CAGRA 索引：
  - 使用方式一：`index_type: "CAGRA"`（GPU 构建与检索，需安装 cuVS/RAFT）
  - 使用方式二：`index_type: "CAGRA->HNSW32,Flat"`（GPU 构建后转换为 CPU HNSW，用于 CPU 检索与可选缓存）
- 依赖与限制（重要）：
  - 环境必须支持 CUDA 12 运行时（例如可加载 `libcudart.so.12`）。
  - 若检测到环境不支持 CUDA 12，将直接禁用 CAGRA 并抛出错误（请改用 HNSW/IVF 等索引）。
  - 需要可用的 CUDA GPU 与 cuVS Python 包（建议通过 RAPIDS/conda 安装）。
  - 在 macOS 或无 GPU 环境下将无法实际运行 CAGRA；适配器会抛出清晰错误信息。
- 参数：
  - 构建参数（`index_param`）：`graph_degree`、`intermediate_graph_degree`、`metric`（`L2`/`IP`）。
  - 搜索参数（`search_param`）：`search_width`、`itopk_size`、`refine_ratio`。
  - 若使用 `CAGRA->HNSW...`，可在 `search_param` 中继续设置 `efSearch` 等 HNSW 搜索参数。

配置示例见 `config.yaml.template` 中的两条 CAGRA 配置。

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
# CPU 模式（输入为数据集目录路径，目录名即数据集名）
python generate_dataset.py -i data/sift -q 1000 -k 100

# GPU 模式：使用 GPU 加速，自动选择批处理大小
python generate_dataset.py -i data/sift -q 1000 -k 100 --gpu

# GPU 模式：自定义批处理大小
python generate_dataset.py -i data/sift -q 1000 -k 100 --gpu --batch-size 500
```

#### 参数说明
- `-i, --input`: 输入的数据集目录路径（目录下需包含同名 `.fvecs` 文件，例如 `data/sift/sift.fvecs`）
- `-q, --queries`: query 集合的向量数量
- `-k, --topk`: 每个 query 的最近邻数量 (默认: 100)
- `--gpu`: 使用 GPU 加速 groundtruth 生成
- `--batch-size`: 批处理大小 (0 表示自动选择，默认: 0)
- `--memory-limit`: 内存限制 (GB，用于自动选择批处理大小，默认: 16.0)
- `--gpu-memory`: GPU 内存大小 (GB，用于可行性检查与分块建议，默认: 10.0)

#### 示例

```bash
# CPU 模式：从 data/sift/sift.fvecs 中提取 1000 个 query，生成 top-100 groundtruth
python generate_dataset.py -i data/sift -q 1000 -k 100

# GPU 模式：使用 GPU 加速，自动选择批处理大小
python generate_dataset.py -i data/sift -q 1000 -k 100 --gpu

# GPU 模式：自定义批处理大小为 500
python generate_dataset.py -i data/sift -q 1000 -k 100 --gpu --batch-size 500

# 大数据集：限制内存使用为 16GB，同时进行 GPU 内存可行性检查（24GB）
python generate_dataset.py -i data/large -q 10000 -k 100 --gpu --memory-limit 16.0 --gpu-memory 24
```

这将生成以下文件（位于数据集目录下）：
- `data/sift/sift_query.fvecs`: 包含 1000 个 query 向量
- `data/sift/sift_base.fvecs`: 包含剩余的 base 向量
- `data/sift/sift_groundtruth.ivecs`: 每个 query 的前 100 个最近邻索引

#### 性能优化建议

1. **GPU 加速**: 对于大规模数据集，强烈建议使用 `--gpu` 参数，可以显著提升处理速度
2. **批处理大小**: 
   - 自动模式 (`--batch-size 0`) 会根据可用内存自动选择合适的批处理大小
   - 手动设置时，建议根据 GPU 内存大小调整：
     - 8GB GPU: 500-1000
     - 16GB GPU: 1000-2000
     - 24GB+ GPU: 2000+
3. **内存管理**: 使用 `--memory-limit` 参数限制内存使用，避免系统内存不足
4. **GPU 内存检查**: 使用 `--gpu-memory` 指定可用 GPU 内存大小，程序会给出是否可行的提示与分块建议

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
