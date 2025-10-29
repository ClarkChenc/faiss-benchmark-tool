import faiss

def create_index(index_type: str, dimension: int, use_gpu: bool = False, params: dict = None):
    """Creates a Faiss index, sets its parameters, and moves it to GPU if requested."""
    try:
        index = faiss.index_factory(dimension, index_type)

        if params:
            param_string = ",".join(f"{k}={v}" for k, v in params.items())
            faiss.ParameterSpace().set_index_parameters(index, param_string)

        if use_gpu:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            return gpu_index
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to create index '{index_type}' with dimension {dimension}: {e}")