import faiss

def _is_cuda12_available() -> bool:
    """Detect whether CUDA 12 runtime is available.

    This checks for common CUDA 12 shared libraries being loadable.
    If detection fails, we conservatively return False.
    """
    try:
        import ctypes
        import os
        import sys
        candidates = [
            "libcuda.so",       # NVIDIA driver stub (version-independent)
            "libcudart.so.12",  # CUDA 12 runtime
        ]
        # Try load from default search paths
        for lib in candidates:
            try:
                ctypes.CDLL(lib)
                # If libcudart.so.12 loads, we consider CUDA12 present
                if lib.endswith(".so.12"):
                    return True
            except Exception:
                continue
        # As a fallback, inspect environment variables that hint CUDA 12
        cuda_version_env = os.environ.get("CUDA_VERSION", "")
        if cuda_version_env.startswith("12"):
            return True
        # Some environments expose CUDA path variables
        for var in ("CUDA_HOME", "CUDA_PATH"):
            p = os.environ.get(var)
            if p and "12" in p:
                return True
    except Exception:
        pass
    return False

def _maybe_create_cagra_adapter(index_type: str, dimension: int, params: dict | None):
    """Create CAGRA adapter only when CUDA12 is available; otherwise raise informative error.

    We avoid importing the adapter unless environment meets the requirement.
    """
    if not _is_cuda12_available():
        raise RuntimeError(
            "CAGRA is disabled: CUDA 12 runtime not detected. "
            "Please upgrade/install CUDA 12 and cuVS/RAFT to use CAGRA, "
            "or switch to other index types (e.g., HNSW/IVF)."
        )
    # Lazy import to prevent pulling cuVS/Faiss GPU deps when not needed
    from .cagra_adapter import CagraIndexAdapter
    # Support optional conversion syntax: "CAGRA->HNSW32,Flat"
    convert_to_hnsw = None
    if "->" in index_type:
        try:
            _, convert_to_hnsw = index_type.split("->", 1)
            convert_to_hnsw = convert_to_hnsw.strip()
        except Exception:
            convert_to_hnsw = None
    index_params = params or {}
    return CagraIndexAdapter(dimension=int(dimension), build_params=index_params, convert_to_hnsw=convert_to_hnsw)

def create_index(index_type: str, dimension: int, use_gpu: bool = False, params: dict = None):
    """Creates a Faiss index with build-time params, moves it to GPU if requested.

    Notes:
    - Only apply build-time params here (e.g., HNSW efConstruction, IVF nlist if provided).
    - Search-time params (e.g., nprobe, efSearch) MUST be applied during search.
    """
    try:
        # 确保 dimension 是标准 Python int 类型
        dimension = int(dimension)
        
        # Route to CAGRA adapter when requested (CUDA12 required)
        if "CAGRA" in index_type.upper():
            return _maybe_create_cagra_adapter(index_type=index_type, dimension=dimension, params=params)

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