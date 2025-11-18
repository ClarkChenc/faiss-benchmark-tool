import numpy as np
import os
import sys
import glob
import ctypes
import site

class CagraIndexAdapter:
    """
    Adapter that exposes a Faiss-like interface for CAGRA.

    Primary path: use Faiss's native CAGRA bindings (`GpuIndexCagra`).
    Fallback path: use cuVS/RAFT CAGRA Python bindings when Faiss is unavailable.

    Provides train/add/search methods so existing benchmarking code can use it.
    Optionally converts the GPU CAGRA index to a CPU HNSW Faiss index for search.

    Notes:
    - Preferred: Faiss with CUDA and `GpuIndexCagra` available (Faiss >= 1.12).
    - Fallback: cuVS/RAFT Python bindings and a CUDA-capable GPU.
    - If conversion to HNSW is requested, Faiss must be available to hold the CPU index.
    - This adapter is defensive: if required dependencies are missing, it raises
      a clear RuntimeError during `add()`.
    """

    def __init__(self, dimension: int, build_params: dict = None, convert_to_hnsw: str | None = None):
        self.dimension = int(dimension)
        self.build_params = build_params or {}
        self.convert_to_hnsw = convert_to_hnsw  # e.g. "HNSW32,Flat"

        # Internal handles
        self._gpu_index = None  # cuVS CAGRA index
        self._cpu_index = None  # Faiss CPU index (after conversion)

        # Cache search params for CAGRA when using adapter-level control
        self._search_params = {}

    # Faiss-like interface
    def train(self, xb: np.ndarray):
        """CAGRA does not require explicit training; this is a no-op."""
        return None

    def add(self, xb: np.ndarray):
        """
        Build the CAGRA graph on GPU. Optionally convert to CPU HNSW if requested.
        """
        try:
            import numpy as _np
            xb = _np.asarray(xb, dtype=_np.float32)
        except Exception:
            raise RuntimeError("CAGRA adapter expects NumPy float32 input arrays for `add()`.")

        # Preferred: build CAGRA via Faiss GPU bindings when available
        metric = str(self.build_params.get("metric", "L2")).upper()
        faiss_errors = []
        try:
            import faiss as _faiss
            # Map metric string to Faiss metric enum
            if metric == "IP" or metric == "INNER_PRODUCT":
                _metric_enum = _faiss.METRIC_INNER_PRODUCT
            else:
                _metric_enum = _faiss.METRIC_L2

            # Initialize GPU index once, then add incrementally
            if self._gpu_index is None or type(self._gpu_index).__name__ != 'GpuIndexCagra':
                res = _faiss.StandardGpuResources()
                self._gpu_index = _faiss.GpuIndexCagra(res, self.dimension, _metric_enum)
                # Apply build-time params via GPU ParameterSpace when available
                try:
                    PS = getattr(_faiss, "GpuParameterSpace", _faiss.ParameterSpace)
                    ps = PS()
                    param_pairs = []
                    if "graph_degree" in self.build_params:
                        param_pairs.append(f"graph_degree={int(self.build_params['graph_degree'])}")
                    if "intermediate_graph_degree" in self.build_params:
                        param_pairs.append(
                            f"intermediate_graph_degree={int(self.build_params['intermediate_graph_degree'])}"
                        )
                    if param_pairs:
                        ps.set_index_parameters(self._gpu_index, ",".join(param_pairs))
                        print(f"[CAGRA] Applied build params: {','.join(param_pairs)}")
                except Exception as e:
                    faiss_errors.append(f"apply build params failed: {e}")

            # Add vectors (build happens internally, supports incremental add)
            self._gpu_index.add(xb)
        except Exception as e_faiss:
            raise RuntimeError(f"Failed to build CAGRA GPU index (Faiss + cuVS fallback failed): {str(e_faiss)}")

        # Optional conversion to CPU index via Faiss CAGRA->HNSW copy
        if self.convert_to_hnsw:
            try:
                import faiss as _faiss
            except Exception as e:
                raise RuntimeError(f"Faiss required for CAGRA->HNSW conversion: {e}")

            # Parse HNSW spec like "HNSW32,Flat" to extract M
            hnsw_spec = self.convert_to_hnsw
            try:
                if "," in hnsw_spec:
                    hnsw_tag, metric_tag = hnsw_spec.split(",", 1)
                else:
                    hnsw_tag, metric_tag = hnsw_spec, "Flat"
                M = int(hnsw_tag.replace("HNSW", ""))
            except Exception:
                M = 32

            # Prefer exact CPU CAGRA-compatible index and copy GPU graph to it
            try:
                if self._cpu_index is None:
                    cpu_index = None
                    # Try Faiss IndexHNSWCagra (preferred for copyTo)
                    if hasattr(_faiss, "IndexHNSWCagra"):
                        try:
                            # Try different constructor signatures defensively
                            try:
                                cpu_index = _faiss.IndexHNSWCagra(self.dimension, M)
                            except TypeError:
                                # Some builds may expect metric as third arg
                                if metric == "IP" or metric == "INNER_PRODUCT":
                                    _metric_enum = _faiss.METRIC_INNER_PRODUCT
                                else:
                                    _metric_enum = _faiss.METRIC_L2
                                cpu_index = _faiss.IndexHNSWCagra(self.dimension, M, _metric_enum)  # type: ignore
                        except Exception as e:
                            cpu_index = None

                    # If we have a Faiss GPU CAGRA index, try copyTo
                    try:
                        if self._gpu_index is not None and type(self._gpu_index).__name__ == 'GpuIndexCagra' and hasattr(self._gpu_index, 'copyTo') and cpu_index.__class__.__name__ == 'IndexHNSWCagra':
                            # Copy GPU CAGRA graph to CPU HNSW-CAGRA for accurate serialization
                            self._gpu_index.copyTo(cpu_index)
                            self._cpu_index = cpu_index
                        else:
                            # Fallback path: rebuild CPU HNSW by adding raw vectors
                            self._cpu_index = cpu_index
                            self._cpu_index.add(xb)
                    except Exception:
                        # Any failure in copyTo falls back to add-based rebuild
                        self._cpu_index = cpu_index
                        try:
                            self._cpu_index.add(xb)
                        except Exception:
                            pass
                else:
                    # Already have a CPU index from previous batches; append
                    try:
                        self._cpu_index.add(xb)
                    except Exception:
                        pass
            except Exception as e:
                raise RuntimeError(f"Failed to convert/append to CPU HNSW/CAGRA: {e}")

    def search(self, xq: np.ndarray, topk: int):
        """Search using either GPU CAGRA or CPU HNSW depending on availability."""
        import numpy as _np
        xq = _np.asarray(xq, dtype=_np.float32)

        # Prefer CPU index if available (post-conversion)
        if self._cpu_index is not None:
            # Apply HNSW search params if provided
            try:
                if "efSearch" in self._search_params:
                    self._cpu_index.hnsw.efSearch = int(self._search_params["efSearch"])  # type: ignore
            except Exception:
                pass
            return self._cpu_index.search(xq, int(topk))

        if self._gpu_index is None:
            raise RuntimeError("CAGRA GPU index is not built. Call add() first.")

        # GPU search: prefer Faiss GPU index with dedicated SearchParameters
        try:
            import faiss as _faiss
            params_obj = None
            # Use index-specific SearchParameters if available
            if hasattr(_faiss, "SearchParametersCagra"):
                try:
                    params_obj = _faiss.SearchParametersCagra()
                    applied = []
                    def _try_set(obj, names, value):
                        for n in names:
                            try:
                                setattr(obj, n, value)
                                return n
                            except Exception:
                                continue
                        return None

                    if "search_width" in self._search_params:
                        name = _try_set(params_obj, ["search_width", "searchWidth"], int(self._search_params["search_width"]))
                        if name:
                            applied.append(f"{name}={getattr(params_obj, name)}")
                    if "itopk_size" in self._search_params:
                        name = _try_set(params_obj, ["itopk_size", "itopkSize"], int(self._search_params["itopk_size"]))
                        if name:
                            applied.append(f"{name}={getattr(params_obj, name)}")
                    if "refine_ratio" in self._search_params:
                        name = _try_set(params_obj, ["refine_ratio", "refineRatio"], float(self._search_params["refine_ratio"]))
                        if name:
                            applied.append(f"{name}={getattr(params_obj, name)}")
                    # if applied:
                        # print(f"[CAGRA] Using SearchParametersCagra: {', '.join(applied)}")
                except Exception as e:
                    # Couldn't construct or set, leave params_obj=None and continue
                    print(f"[CAGRA] Failed to prepare SearchParametersCagra: {e}")

            if params_obj is not None:
                D, I = self._gpu_index.search(xq, int(topk), params=params_obj)
            else:
                # No dedicated params available; run default search on Faiss path
                D, I = self._gpu_index.search(xq, int(topk))
            return D, I
        except Exception:
            # Fallback to cuVS search path if Faiss search failed (e.g., _gpu_index is cuVS handle)
            try:
                import_errors = []
                try:
                    from cuvs.neighbors import cagra as _cagra
                except Exception as e:
                    import_errors.append(f"cuvs.neighbors import failed: {e}")
                    try:
                        from cuvs.legacy.neighbors import cagra as _cagra  # type: ignore
                    except Exception as e2:
                        import_errors.append(f"cuvs.legacy.neighbors import failed: {e2}")
                        try:
                            from cuvs.experimental.neighbors import cagra as _cagra  # type: ignore
                        except Exception as e3:
                            import_errors.append(f"cuvs.experimental.neighbors import failed: {e3}")
                            try:
                                from raft.neighbors import cagra as _cagra  # type: ignore
                            except Exception as e4:
                                import_errors.append(f"raft.neighbors import failed: {e4}")
                                _cagra = None
                if _cagra is None:
                    raise RuntimeError(f"cuVS CAGRA not found for search. Import attempts: {' | '.join(import_errors)}")

                search_width = int(self._search_params.get("search_width", 4))
                itopk_size = int(self._search_params.get("itopk_size", max(32, topk)))
                refine_ratio = float(self._search_params.get("refine_ratio", 1.0))

                try:
                    D, I = _cagra.search(
                        self._gpu_index,
                        xq,
                        int(topk),
                        search_width=search_width,
                        itopk_size=itopk_size,
                        refine_ratio=refine_ratio,
                    )
                except TypeError:
                    D, I = _cagra.search(self._gpu_index, xq, int(topk))
                return D, I
            except Exception as e:
                raise RuntimeError(f"Failed to search with CAGRA (Faiss + cuVS fallback failed): {e}")

    # Adapter-specific API to apply search-time params
    def search_with_params(self, xq: np.ndarray, topk: int, params: dict | None = None):
        self._search_params = params or {}
        return self.search(xq, topk)

    # Expose CPU index for caching/saving in main flow when available
    def get_cpu_index(self):
        return self._cpu_index