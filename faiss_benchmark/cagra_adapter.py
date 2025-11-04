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
                # Apply build-time params via ParameterSpace if supported
                ps = _faiss.ParameterSpace()
                try:
                    if "graph_degree" in self.build_params:
                        ps.set_index_parameter(self._gpu_index, "graph_degree", int(self.build_params["graph_degree"]))
                except Exception as e:
                    faiss_errors.append(f"set graph_degree failed: {e}")
                try:
                    if "intermediate_graph_degree" in self.build_params:
                        ps.set_index_parameter(self._gpu_index, "intermediate_graph_degree", int(self.build_params["intermediate_graph_degree"]))
                except Exception as e:
                    faiss_errors.append(f"set intermediate_graph_degree failed: {e}")

            # Add vectors (build happens internally, supports incremental add)
            self._gpu_index.add(xb)
        except Exception as e_faiss:
            faiss_errors.append(str(e_faiss))
            # Fallback: build via cuVS bindings
            try:
                # Attempt to import cuVS CAGRA
                import_errors = []
                try:
                    from cuvs.neighbors import cagra as _cagra
                except Exception as e:
                    import_errors.append(f"cuvs.neighbors import failed: {e}")
                    # Alternative import paths (older RAFT/cuVS layouts)
                    try:
                        from cuvs.legacy.neighbors import cagra as _cagra  # type: ignore
                    except Exception as e2:
                        import_errors.append(f"cuvs.legacy.neighbors import failed: {e2}")
                        # Try experimental or raft paths as additional fallbacks
                        try:
                            from cuvs.experimental.neighbors import cagra as _cagra  # type: ignore
                        except Exception as e3:
                            import_errors.append(f"cuvs.experimental.neighbors import failed: {e3}")
                            try:
                                from raft.neighbors import cagra as _cagra  # type: ignore
                            except Exception as e4:
                                import_errors.append(f"raft.neighbors import failed: {e4}")
                                _cagra = None

                # If import failed due to missing shared library, try to locate and pre-load it
                if _cagra is None and any("libcuvs_c.so" in str(err) for err in import_errors):
                    candidates = []
                    # Search common site-packages and sys.path locations for libcuvs_c.so
                    paths = []
                    try:
                        paths.extend(site.getsitepackages())
                    except Exception:
                        pass
                    try:
                        paths.append(site.getusersitepackages())
                    except Exception:
                        pass
                    paths.extend([p for p in sys.path if isinstance(p, str)])
                    for base in paths:
                        try:
                            candidates.extend(glob.glob(os.path.join(base, "**", "libcuvs_c.so"), recursive=True))
                        except Exception:
                            pass
                    # Preload the first matching library, then retry import
                    for libpath in candidates:
                        try:
                            ctypes.CDLL(libpath)
                            # Retry import
                            from cuvs.neighbors import cagra as _cagra  # type: ignore
                            break
                        except Exception:
                            continue

                if _cagra is None:
                    # Attach detailed import errors to help diagnose environment issues
                    raise RuntimeError(
                        "cuVS CAGRA not found. Please install RAPIDS/cuVS with CUDA. "
                        "Refer to https://rapids.ai/install for conda installation instructions. "
                        f"Import attempts: {' | '.join(import_errors)} | faiss errors: {' | '.join(faiss_errors)}"
                    )

                # Build params with sane fallbacks
                graph_degree = int(self.build_params.get("graph_degree", 32))
                intermediate_graph_degree = int(self.build_params.get("intermediate_graph_degree", max(64, 2 * graph_degree)))

                # Build GPU index via cuVS (no incremental add guaranteed)
                try:
                    self._gpu_index = _cagra.build(
                        xb,
                        graph_degree=graph_degree,
                        intermediate_graph_degree=intermediate_graph_degree,
                        metric=metric,
                    )
                except TypeError:
                    self._gpu_index = _cagra.build(xb)
            except Exception as e:
                raise RuntimeError(f"Failed to build CAGRA GPU index (Faiss + cuVS fallback failed): {e}")

        # Optional conversion to CPU HNSW (incremental-friendly)
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

            # Create or reuse CPU HNSW index and add incrementally
            try:
                if self._cpu_index is None:
                    hnsw_index = _faiss.IndexHNSWFlat(self.dimension, M)
                    hnsw_index.hnsw.efConstruction = int(self.build_params.get("efConstruction", 40))
                    self._cpu_index = hnsw_index
                self._cpu_index.add(xb)
            except Exception as e:
                raise RuntimeError(f"Failed to convert/append to CPU HNSW: {e}")

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

        # GPU search: prefer Faiss GPU index if applicable
        try:
            import faiss as _faiss
            ps = _faiss.ParameterSpace()
            # Apply search-time params if provided and supported
            try:
                if "search_width" in self._search_params:
                    ps.set_index_parameter(self._gpu_index, "search_width", int(self._search_params["search_width"]))
            except Exception:
                pass
            try:
                if "itopk_size" in self._search_params:
                    ps.set_index_parameter(self._gpu_index, "itopk_size", int(self._search_params["itopk_size"]))
            except Exception:
                pass
            try:
                if "refine_ratio" in self._search_params:
                    ps.set_index_parameter(self._gpu_index, "refine_ratio", float(self._search_params["refine_ratio"]))
            except Exception:
                pass
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