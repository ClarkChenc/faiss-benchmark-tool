import numpy as np
import yaml
import os
import shutil

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)

def get_fvecs_info(fname):
    """
    获取 .fvecs 文件的基本信息，不加载数据
    
    Args:
        fname: 文件路径
        
    Returns:
        tuple: (向量数量, 维度)
    """
    with open(fname, 'rb') as f:
        # 读取第一个向量的维度信息
        d = np.frombuffer(f.read(4), dtype='int32')[0]
        
        # 计算文件大小和向量数量
        f.seek(0, 2)  # 移动到文件末尾
        file_size = f.tell()
        vector_size = (d + 1) * 4  # 每个向量的字节数 (维度 + 数据)
        num_vectors = file_size // vector_size
        
    return num_vectors, d

def fvecs_read_range(fname, start_idx, count):
    """
    读取 .fvecs 文件中指定范围的向量
    
    Args:
        fname: 文件路径
        start_idx: 起始向量索引
        count: 读取向量数量
        
    Returns:
        numpy.ndarray: 读取的向量数组
    """
    with open(fname, 'rb') as f:
        # 读取第一个向量的维度信息
        d = np.frombuffer(f.read(4), dtype='int32')[0]
        f.seek(0)  # 重置到文件开头
        
        # 计算偏移量
        vector_size = (d + 1) * 4  # 每个向量的字节数
        offset = start_idx * vector_size
        f.seek(offset)
        
        # 读取指定数量的向量
        data_size = count * vector_size
        data = np.frombuffer(f.read(data_size), dtype='int32')
        
        # 重塑并移除维度列
        vectors = data.reshape(-1, d + 1)[:, 1:].copy()
        
    return vectors.view('float32')

def fvecs_write_streaming(fname, vectors_generator, total_count=None):
    """
    流式写入 .fvecs 文件
    
    Args:
        fname: 输出文件路径
        vectors_generator: 向量生成器，每次产生一批向量
        total_count: 总向量数量（可选，用于显示进度）
    """
    with open(fname, 'wb') as f:
        written_count = 0
        
        for vectors in vectors_generator:
            if len(vectors) == 0:
                continue
                
            # 转换为正确的格式并写入
            vectors = vectors.astype('float32')
            n, d = vectors.shape
            
            # 创建带维度信息的数组
            m1 = np.empty((n, d + 1), dtype='int32')
            m1[:, 0] = d
            m1[:, 1:] = vectors.view('int32')
            
            # 写入文件
            m1.tofile(f)
            written_count += n
            
            if total_count:
                print(f"\r已写入 {written_count}/{total_count} 个向量", end='', flush=True)
        
        if total_count:
            print()  # 换行

def load_config(config_path):
    """
    加载配置文件。如果配置文件不存在，会尝试从模板文件创建。
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    # 如果配置文件不存在，尝试从模板创建
    if not os.path.exists(config_path):
        template_path = config_path + ".template"
        
        if os.path.exists(template_path):
            print(f"配置文件 {config_path} 不存在，正在从模板 {template_path} 创建...")
            shutil.copy2(template_path, config_path)
            print(f"已创建配置文件 {config_path}，请根据需要修改参数。")
        else:
            raise FileNotFoundError(
                f"配置文件 {config_path} 和模板文件 {template_path} 都不存在。"
                f"请确保至少有一个文件存在。"
            )
    
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config