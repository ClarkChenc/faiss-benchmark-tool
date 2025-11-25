import os
import json
import faiss


def _normalize_build_params(build_params: dict | None) -> str:
    """Create a stable string from build params to form cache key."""
    if not build_params:
        return ""
    items = sorted((str(k), str(v)) for k, v in build_params.items())
    return "_".join([f"{k}{v}" for k, v in items])


def _paths(cache_dir: str, dataset: str, index_type: str, build_params: dict | None):
    """Return index and meta paths based on index type and params.

    - CPU-only ScaNN uses `.scann` for index binary
    - Faiss CPU indices use `.index`
    """
    bp = _normalize_build_params(build_params)
    base = f"{dataset}_{index_type}_{bp}" if bp else f"{dataset}_{index_type}"
    if "SCANN" in index_type.upper():
        idx = os.path.join(cache_dir, base + ".scann")
    else:
        idx = os.path.join(cache_dir, base + ".index")
    meta = os.path.join(cache_dir, base + "_meta.json")
    return idx, meta


def ensure_dir(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)


def load(cache_dir: str, dataset: str, index_type: str, build_params: dict | None):
    """Load cached index and metadata.

    Returns (index_object, metadata) or (None, None) if not found.
    ScaNN is returned as ScannIndexAdapter to unify search interface.
    """
    ensure_dir(cache_dir)
    idx_path, meta_path = _paths(cache_dir, dataset, index_type, build_params)
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return None, None
    # Load metadata first
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    if "SCANN" in index_type.upper():
        try:
            from .scann_adapter import ScannIndexAdapter
            idx = ScannIndexAdapter.load_from_cache(idx_path)
            return idx, meta
        except Exception:
            return None, None
    # Faiss CPU index
    try:
        idx = faiss.read_index(idx_path)
        return idx, meta
    except Exception:
        return None, None


def save(cache_dir: str, dataset: str, index_type: str, build_params: dict | None, index_object, metadata: dict | None):
    """Save index and metadata to cache.

    - Only CPU indices are cached (GPU indices must be converted to CPU before saving).
    - For CAGRA adapters, will attempt to extract CPU index via `get_cpu_index`.
    - ScaNN adapters use their internal save method.
    """
    ensure_dir(cache_dir)
    idx_path, meta_path = _paths(cache_dir, dataset, index_type, build_params)

    # Determine what to save for Faiss-like indices
    obj_to_save = index_object
    try:
        # If adapter provides CPU conversion, prefer that
        if hasattr(index_object, 'get_cpu_index'):
            cpu_idx = index_object.get_cpu_index()
            if cpu_idx is not None:
                obj_to_save = cpu_idx
    except Exception:
        pass

    if "SCANN" in index_type.upper():
        # Expect ScannIndexAdapter with save_to_cache
        try:
            if hasattr(index_object, 'save_to_cache'):
                index_object.save_to_cache(idx_path)
            else:
                # Best effort: try direct scann pybind API if exposed
                import scann
                try:
                    scann.scann_ops_pybind.save_searcher(index_object, idx_path)  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"ScaNN save failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to save ScaNN index: {e}")
    else:
        # Faiss CPU index
        try:
            faiss.write_index(obj_to_save, idx_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save Faiss index: {e}")

    # Save metadata
    try:
        with open(meta_path, 'w') as f:
            json.dump(metadata or {}, f)
    except Exception:
        pass

