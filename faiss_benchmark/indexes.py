import faiss

def create_index(index_type: str, dimension: int, use_gpu: bool = False, params: dict = None):
    """Creates a Faiss index with build-time params, moves it to GPU if requested.

    Notes:
    - Only apply build-time params here (e.g., HNSW efConstruction, IVF nlist if provided).
    - Search-time params (e.g., nprobe, efSearch) MUST be applied during search.
    """
    try:
        # 确保 dimension 是标准 Python int 类型
        dimension = int(dimension)
        
        index = faiss.index_factory(dimension, index_type)

        index_params = params or {}

        # Handle HNSW build params
        if "HNSW" in index_type:
            try:
                hnsw_index = faiss.downcast_index(index)
                if hnsw_index and "efConstruction" in index_params:
                    hnsw_index.hnsw.efConstruction = int(index_params["efConstruction"])
            except Exception:
                pass

        # If IVF build param 'nlist' is explicitly provided (index_type may already encode it)
        if "IVF" in index_type and "nlist" in index_params:
            try:
                faiss.ParameterSpace().set_index_parameter(index, "nlist", int(index_params["nlist"]))
            except Exception:
                pass

        if use_gpu:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            return gpu_index
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to create index '{index_type}' with dimension {dimension}: {e}")