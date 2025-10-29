import numpy as np
import os

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_dataset(name):
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