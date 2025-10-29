import os
import numpy as np
from .utils import fvecs_read, ivecs_read

def load_dataset(name):
    """加载完整数据集到内存"""
    data_dir = os.path.join("data", name)
    base_path = os.path.join(data_dir, f"{name}_base.fvecs")
    query_path = os.path.join(data_dir, f"{name}_query.fvecs")
    groundtruth_path = os.path.join(data_dir, f"{name}_groundtruth.ivecs")

    if not all(os.path.exists(p) for p in [base_path, query_path, groundtruth_path]):
        raise FileNotFoundError(f"Dataset files for '{name}' not found in '{data_dir}'. "
                              f"Please download them and place them in the '{data_dir}' directory.")

    xb = fvecs_read(base_path)
    xq = fvecs_read(query_path)
    gt = ivecs_read(groundtruth_path)

    return xb, xq, gt

def get_dataset_info(name):
    """获取数据集信息而不加载数据"""
    data_dir = os.path.join("data", name)
    base_path = os.path.join(data_dir, f"{name}_base.fvecs")
    query_path = os.path.join(data_dir, f"{name}_query.fvecs")
    groundtruth_path = os.path.join(data_dir, f"{name}_groundtruth.ivecs")

    if not all(os.path.exists(p) for p in [base_path, query_path, groundtruth_path]):
        raise FileNotFoundError(f"Dataset files for '{name}' not found in '{data_dir}'. "
                              f"Please download them and place them in the '{data_dir}' directory.")

    # 读取文件头部信息获取维度和数量
    with open(base_path, 'rb') as f:
        d = np.fromfile(f, dtype='int32', count=1)[0]
        f.seek(0, 2)  # 移动到文件末尾
        file_size = f.tell()
        n = file_size // ((d + 1) * 4)  # 每个向量占用 (d+1)*4 字节
    
    # 查询集信息
    xq = fvecs_read(query_path)
    gt = ivecs_read(groundtruth_path)
    
    return {
        'base_count': n,
        'dimension': d,
        'query_count': xq.shape[0],
        'base_path': base_path,
        'query_vectors': xq,
        'groundtruth': gt
    }

def load_base_vectors_batch(base_path, start_idx, batch_size, dimension):
    """分批加载基础向量"""
    with open(base_path, 'rb') as f:
        # 跳到指定位置
        f.seek(start_idx * (dimension + 1) * 4)
        
        # 读取批次数据
        data = np.fromfile(f, dtype='int32', count=batch_size * (dimension + 1))
        
        if len(data) == 0:
            return None
        
        # 重新整形并移除维度信息
        actual_batch_size = len(data) // (dimension + 1)
        if actual_batch_size == 0:
            return None
            
        vectors = data.reshape(actual_batch_size, dimension + 1)[:, 1:].copy()
        return vectors.view('float32')