# generate_dataset.py 使用说明

数据集生成工具：将单个 `.fvecs` 文件分割为 `query`/`base` 两部分，并使用 Faiss 暴力搜索生成准确的 `groundtruth` 文件。支持 GPU 加速、分批处理与内存可行性检查，适用于大规模数据集。

## 目录结构要求

- 输入为“数据集目录路径”，目录名即数据集名；目录中需包含同名 `.fvecs` 文件。
- 示例：`data/sift/sift.fvecs`

运行后会在该目录下生成：
- `data/<name>/<name>_query.fvecs`
- `data/<name>/<name>_base.fvecs`
- `data/<name>/<name>_groundtruth.ivecs`

## 命令行用法

```bash
# CPU 模式（自动建议批处理大小）
python generate_dataset.py -i data/sift -q 1000 -k 100

# GPU 模式（自动建议批处理大小）
python generate_dataset.py -i data/sift -q 1000 -k 100 --gpu

# GPU 模式（自定义批处理大小）
python generate_dataset.py -i data/sift -q 1000 -k 100 --gpu --batch-size 500

# 指定内存限制与 GPU 内存进行可行性检查
python generate_dataset.py -i data/large -q 10000 -k 100 --gpu --memory-limit 16.0 --gpu-memory 24
```

## 参数说明

- `-i, --input`：输入的数据集目录路径（目录下需包含同名 `.fvecs` 文件）
- `-q, --queries`：query 向量数量（将从 `.fvecs` 前部截取）
- `-k, --topk`：每个 query 的最近邻数量（默认 `100`）
- `--gpu`：启用 GPU 加速（需要可用的 CUDA GPU 与 `faiss-gpu`）
- `--batch-size`：批处理大小；`0` 表示自动选择（默认 `0`）
- `--memory-limit`：CPU 内存限制（GB，辅助自动选择批处理大小，默认 `16.0`）
- `--gpu-memory`：GPU 内存大小（GB，用于可行性检查与分块建议，默认 `10.0`）

## 输出文件说明

- `<name>_query.fvecs`：包含 `q` 个 query 向量（从原始 `.fvecs` 前部截取）
- `<name>_base.fvecs`：包含剩余的 base 向量（用于建立索引）
- `<name>_groundtruth.ivecs`：每个 query 的前 `k` 个最近邻索引（int32）

## 运行日志与提示

- 程序会显示数据集统计信息、内存估算、批处理大小选择与进度条。
- 在 `--gpu` 模式下，会进行 GPU 内存可行性检查；若内存不足，会给出分块建议与风险提示。

## 常见问题

- 报错“输入数据不存在”：确认 `-i` 指向数据集目录，且目录下存在同名 `.fvecs` 文件，例如 `data/sift/sift.fvecs`。
- GPU 不可用：确认安装 `faiss-gpu`，且 `faiss.get_num_gpus() > 0`。
- 内存不足：降低 `--batch-size`，或增加 `--memory-limit`/`--gpu-memory`，或改用 CPU 模式。

## 依赖

- `faiss-gpu`（若使用 GPU）
- `numpy`
- `tqdm`