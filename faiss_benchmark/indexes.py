import faiss

def create_index(index_type: str, dimension: int, use_gpu: bool = False, params: dict = None):
    """Creates a Faiss index, sets its parameters, and moves it to GPU if requested."""
    try:
        # 确保 dimension 是标准 Python int 类型
        dimension = int(dimension)
        
        index = faiss.index_factory(dimension, index_type)

        if params:
            # Handle IVF params
            if "nprobe" in params:
                faiss.ParameterSpace().set_index_parameter(index, "nprobe", params["nprobe"])

            # Handle HNSW params
            if "HNSW" in index_type:
                hnsw_index = faiss.downcast_index(index)
                if hnsw_index and "efConstruction" in params:
                    hnsw_index.hnsw.efConstruction = params["efConstruction"]

        if use_gpu:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            return gpu_index
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to create index '{index_type}' with dimension {dimension}: {e}")