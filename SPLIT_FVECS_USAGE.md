# .fvecs 数据分割工具使用说明

## 概述

`split_fvecs.py` 是一个专门用于分割大型 `.fvecs` 向量文件的工具。它支持内存优化的流式处理，可以将大文件分割成指定大小的多个小文件，避免内存溢出问题。

## 主要功能

- ✅ **内存优化**: 使用流式处理，支持处理任意大小的文件
- ✅ **灵活分割**: 支持自定义分割大小和起始位置
- ✅ **智能建议**: 根据内存限制自动建议最优分割大小
- ✅ **完整性验证**: 自动验证分割结果的正确性
- ✅ **进度显示**: 实时显示分割进度和详细信息
- ✅ **错误处理**: 完善的错误检查和异常处理

## 安装要求

确保已安装项目依赖：
```bash
pip install -r requirements.txt
```

## 基本用法

### 1. 查看文件信息
```bash
python split_fvecs.py -i data/large.fvecs --info-only
```

### 2. 基本分割
```bash
python split_fvecs.py -i data/large.fvecs -o data/splits -s 1000000
```

### 3. 自动建议分割大小
```bash
python split_fvecs.py -i data/large.fvecs -o data/splits --suggest-size --memory 8
```

### 4. 自定义前缀和起始位置
```bash
python split_fvecs.py -i data/sift.fvecs -o data/splits -s 500000 -p sift_split --start 2000000
```

## 命令行参数

| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `-i, --input` | 输入的 .fvecs 文件路径 | ✅ | - |
| `-o, --output` | 输出目录路径 | ❌ | - |
| `-s, --size` | 每个分割文件的向量数量 | ❌ | - |
| `-p, --prefix` | 输出文件名前缀 | ❌ | 输入文件名 |
| `--start` | 起始向量索引 | ❌ | 0 |
| `--suggest-size` | 根据内存限制建议分割大小 | ❌ | false |
| `--memory` | 内存限制 (GB) | ❌ | 4.0 |
| `--info-only` | 仅显示文件信息 | ❌ | false |

## 使用示例

### 示例 1: 处理大型 SIFT 数据集
```bash
# 查看文件信息
python split_fvecs.py -i data/sift1b_base.fvecs --info-only

# 根据 16GB 内存限制建议分割大小
python split_fvecs.py -i data/sift1b_base.fvecs --suggest-size --memory 16

# 执行分割
python split_fvecs.py -i data/sift1b_base.fvecs -o data/sift_splits -s 10000000 -p sift1b
```

### 示例 2: 部分数据分割
```bash
# 从第 500 万个向量开始，分割 1000 万个向量
python split_fvecs.py -i data/large.fvecs -o data/partial_splits -s 2000000 --start 5000000
```

### 示例 3: 小内存环境处理
```bash
# 在 4GB 内存限制下处理大文件
python split_fvecs.py -i data/large.fvecs -o data/small_splits --suggest-size --memory 4
```

## 输出文件格式

分割后的文件命名格式：`{prefix}_part_{序号}.fvecs`

例如：
- `large_test_part_0001.fvecs`
- `large_test_part_0002.fvecs`
- `sift_split_part_0001.fvecs`

## 内存使用估算

工具会自动估算内存使用量：

- **单块内存需求** = 分割大小 × 向量维度 × 4 字节
- **缓冲区内存** = 批处理大小 × 向量维度 × 4 字节  
- **总内存需求** = 单块内存 + 缓冲区内存 + 系统开销

## 性能优化建议

### 1. 选择合适的分割大小
- **小文件**: 便于处理，但文件数量多
- **大文件**: 文件数量少，但内存需求高
- **建议**: 使用 `--suggest-size` 自动优化

### 2. 内存限制设置
- **保守设置**: 设置为实际内存的 70-80%
- **示例**: 16GB 内存设置为 12-13GB

### 3. 存储空间考虑
- 确保输出目录有足够空间
- 分割不会压缩数据，总大小不变

## 错误处理

### 常见错误及解决方案

1. **文件不存在**
   ```
   ❌ 错误: 输入文件不存在
   ```
   检查文件路径是否正确

2. **内存不足**
   ```
   ⚠️ 警告: 估算内存需求超过限制
   ```
   减小分割大小或增加内存限制

3. **磁盘空间不足**
   ```
   ❌ 分割过程中出现错误: No space left on device
   ```
   清理磁盘空间或选择其他输出目录

4. **权限问题**
   ```
   ❌ 分割过程中出现错误: Permission denied
   ```
   检查输出目录的写入权限

## 验证分割结果

工具会自动验证分割结果：

- ✅ **向量数量**: 确保总向量数匹配
- ✅ **维度一致**: 确保所有文件维度相同
- ✅ **文件完整**: 检查每个文件的完整性

## 与其他工具集成

### 1. 与 generate_dataset.py 配合使用
```bash
# 先分割大文件
python split_fvecs.py -i data/large.fvecs -o data/splits -s 1000000

# 然后为每个分割文件建索引
for file in data/splits/*.fvecs; do
    python generate_dataset.py -d "$file" -o "indices/$(basename $file .fvecs)"
done
```

### 2. 批量处理
```bash
# 处理多个文件
for file in data/*.fvecs; do
    python split_fvecs.py -i "$file" -o "splits/$(basename $file .fvecs)" -s 500000
done
```

## 技术细节

### 内存优化策略
- 使用生成器模式避免一次性加载大文件
- 批量读取（默认 50,000 向量/批次）
- 流式写入，减少内存峰值

### 文件格式兼容性
- 完全兼容标准 .fvecs 格式
- 支持任意维度的向量数据
- 保持原始数据精度（float32）

### 性能特点
- 处理速度: ~20-40 文件/秒（取决于文件大小）
- 内存占用: 通常 < 200MB（不依赖于文件大小）
- 支持文件大小: 理论上无限制

## 故障排除

如果遇到问题，请按以下步骤排查：

1. **检查文件格式**: 确保输入文件是有效的 .fvecs 格式
2. **验证路径**: 确保输入文件存在，输出目录可写
3. **检查内存**: 使用 `--suggest-size` 获取合适的分割大小
4. **查看日志**: 注意工具输出的错误信息和警告
5. **测试小文件**: 先用小文件测试工具功能

如需更多帮助，请查看项目文档或提交 issue。