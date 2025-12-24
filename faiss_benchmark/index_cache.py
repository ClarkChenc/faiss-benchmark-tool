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
    - hnswlib CPU indices use `.hnswlib`
    - Faiss CPU indices use `.index`
    """
    bp = _normalize_build_params(build_params)
    base = f"{dataset}_{index_type}_{bp}" if bp else f"{dataset}_{index_type}"
    if "SCANN" in index_type.upper():
        from importlib.metadata import version
        idx = os.path.join(cache_dir, base + ".scann_" + version("scann"))
        meta = os.path.join(cache_dir, base + ".scann_" + version("scann") + "_meta.json")
    elif "HNSWLIB" in index_type.upper():
        idx = os.path.join(cache_dir, base + ".hnswlib")
        meta = os.path.join(cache_dir, base + "_meta.json")
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
    if "HNSWLIB_SPLIT" in index_type.upper():
        dim = int(meta.get("dimension", 0))
        space = str(meta.get("space", "l2"))
        seg_num = int(meta.get("seg_num", 0))
        segment_sizes = list(meta.get("segment_sizes", []))
        is_merge = bool(meta.get("is_merge", False))
        if dim <= 0 or seg_num <= 0 or len(segment_sizes) == 0:
            return None, None
        try:
            from .hnswlib_adapter import HnswlibSplitIndexAdapter
            num_threads = os.environ.get("OMP_NUM_THREADS", "1")
            idx = HnswlibSplitIndexAdapter.load_from_cache(
                idx_path, 
                dimension=dim, 
                space=space, 
                seg_num=seg_num, 
                segment_sizes=segment_sizes, 
                num_threads=int(num_threads),
                is_merge=is_merge,
                build_params=build_params
            )
            return idx, meta
        except Exception:
            return None, None
    if "HNSWLIB" in index_type.upper():
        # Expect metadata to contain dimension, space, and max_elements
        dim = int(meta.get("dimension", 0))
        space = str(meta.get("space", "l2"))
        max_elems = int(meta.get("num_elements", 0))
        if dim <= 0 or max_elems <= 0:
            return None, None
        try:
            from .hnswlib_adapter import HnswlibIndexAdapter
            num_threads = os.environ.get("OMP_NUM_THREADS", "1")
            idx = HnswlibIndexAdapter.load_from_cache(idx_path, dimension=dim, space=space, max_elements=max_elems, num_threads=int(num_threads))
            return idx, meta
        except Exception as e:
            print(f"{e}")
            return None, None

    if "FastScan" in index_type:
        try:
            from .fastscan_ivfpq_adapter import FastScanIVFPQAdapter
            dim = int(meta.get("dimension", 0))

            idx = FastScanIVFPQAdapter.load_from_cache(idx_path, dimension = dim, build_params=build_params)
            return idx, meta
        except Exception:
            print(f"load fastscan cache failed")
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
                raise RuntimeError(f"scann adapter not has save_to_cache method")
        except Exception as e:
            raise RuntimeError(f"Failed to save ScaNN index: {e}")
    elif "HNSWLIB" in index_type.upper():
        # hnswlib adapter must provide save_to_cache
        try:
            if hasattr(index_object, 'save_to_cache'):
                index_object.save_to_cache(idx_path)
            else:
                raise RuntimeError("HNSWLIB index object does not support save_to_cache")
        except Exception as e:
            raise RuntimeError(f"Failed to save hnswlib index: {e}")
    else:
        # Faiss CPU index or adapter wrapping a Faiss index
        try:
            # Prefer adapter's save_to_cache when available
            if hasattr(index_object, 'save_to_cache'):
                index_object.save_to_cache(idx_path)
            else:
                # Try to extract underlying index
                underlying = None
                try:
                    if hasattr(index_object, 'get_cpu_index'):
                        underlying = index_object.get_cpu_index()
                    elif hasattr(index_object, '_index'):
                        underlying = getattr(index_object, '_index')
                except Exception:
                    underlying = None
                target = underlying or obj_to_save
                # If target is a GPU index, convert to CPU before saving
                try:
                    target = faiss.index_gpu_to_cpu(target)
                except Exception:
                    pass
                # Ensure we have a valid Faiss Index instance
                try:
                    faiss.write_index(target, idx_path)
                except TypeError:
                    raise RuntimeError("write_index 期望 Faiss Index 对象和字符串路径，请检查传入对象类型")
        except Exception as e:
            raise RuntimeError(f"Failed to save Faiss index: {e}")

    # Save metadata
    try:
        meta_out = dict(metadata or {})
        # Enrich metadata for loaders that require extra info
        if "HNSWLIB_SPLIT" in index_type.upper():
            try:
                dim = getattr(index_object, 'dimension', None)
            except Exception:
                dim = None
            try:
                space = getattr(index_object, 'space', None)
            except Exception:
                space = None
            num_elems = None
            try:
                if hasattr(index_object, '_added_total'):
                    num_elems = int(getattr(index_object, '_added_total'))
            except Exception:
                num_elems = None
            seg_num = None
            seg_sizes = []
            try:
                if hasattr(index_object, '_segments'):
                    seg_num = len(getattr(index_object, '_segments'))
                    seg_sizes = [int(seg.get("added", 0)) for seg in getattr(index_object, '_segments')]
            except Exception:
                pass
            if dim is not None:
                meta_out['dimension'] = int(dim)
            if space is not None:
                meta_out['space'] = str(space)
            if num_elems is not None:
                meta_out['num_elements'] = int(num_elems)
            if seg_num is not None and seg_num > 0:
                meta_out['seg_num'] = int(seg_num)
            if seg_sizes:
                meta_out['segment_sizes'] = list(seg_sizes)
            try:
                is_merge = getattr(index_object, 'is_merge', False)
                meta_out['is_merge'] = bool(is_merge)
            except Exception:
                pass
        elif "HNSWLIB" in index_type.upper():
            # ensure dimension and space and num_elements are present
            try:
                dim = getattr(index_object, 'dimension', None)
            except Exception:
                dim = None
            try:
                space = getattr(index_object, 'space', None)
            except Exception:
                space = None
            # number of elements added
            num_elems = None
            try:
                if hasattr(index_object, '_added'):
                    num_elems = int(getattr(index_object, '_added'))
            except Exception:
                num_elems = None
            if dim is not None:
                meta_out['dimension'] = int(dim)
            if space is not None:
                meta_out['space'] = str(space)
            if num_elems is not None:
                meta_out['num_elements'] = int(num_elems)
        with open(meta_path, 'w') as f:
            json.dump(meta_out, f)
    except Exception:
        pass
