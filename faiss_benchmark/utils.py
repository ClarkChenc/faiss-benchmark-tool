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