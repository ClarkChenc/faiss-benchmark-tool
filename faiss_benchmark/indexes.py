import faiss

def create_index(index_type: str, dimension: int, use_gpu: bool = False):
    """Creates a Faiss index based on the specified type and moves it to GPU if requested."""
    try:
        index = faiss.index_factory(dimension, index_type)
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        return index
    except Exception as e:
        print(f"Error creating index of type {index_type}: {e}")
        return None