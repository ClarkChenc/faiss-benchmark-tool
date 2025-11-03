import numpy as np

class CagraIndexAdapter:
    """
    Adapter for cuVS CAGRA index to present a Faiss-like interface.

    Provides train/add/search methods so existing benchmarking code can use it.
    Optionally converts the GPU CAGRA index to a CPU HNSW Faiss index for search.

    Notes:
    - Requires `cuvs` (or RAFT/cuVS Python bindings) and a CUDA-capable GPU for CAGRA.
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

        # Build CAGRA on GPU
        try:
            # Attempt to import cuVS CAGRA
            try:
                from cuvs.neighbors import cagra as _cagra
            except Exception:
                # Alternative import paths (older RAFT/cuVS layouts)
                try:
                    from cuvs.legacy.neighbors import cagra as _cagra  # type: ignore
                except Exception:
                    _cagra = None

            if _cagra is None:
                raise RuntimeError(
                    "cuVS CAGRA not found. Please install RAPIDS/cuVS with CUDA. "
                    "Refer to https://rapids.ai/install for conda installation instructions."
                )

            # Build params with sane fallbacks
            graph_degree = int(self.build_params.get("graph_degree", 32))
            intermediate_graph_degree = int(self.build_params.get("intermediate_graph_degree", max(64, 2 * graph_degree)))
            metric = str(self.build_params.get("metric", "L2")).upper()

            # Build GPU index
            self._gpu_index = _cagra.build(
                xb,
                graph_degree=graph_degree,
                intermediate_graph_degree=intermediate_graph_degree,
                metric=metric,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to build CAGRA GPU index: {e}")

        # Optional conversion to CPU HNSW
        if self.convert_to_hnsw:
            try:
                import faiss as _faiss
            except Exception as e:
                raise RuntimeError(f"Faiss required for CAGRA->HNSW conversion: {e}")

            # Parse HNSW spec like "HNSW32,Flat" to extract M and metric
            hnsw_spec = self.convert_to_hnsw
            try:
                # Extract HNSW M from string (e.g., HNSW32 -> M=32)
                if "," in hnsw_spec:
                    hnsw_tag, metric_tag = hnsw_spec.split(",", 1)
                else:
                    hnsw_tag, metric_tag = hnsw_spec, "Flat"
                # default metric for Faiss: Flat implies L2 on IndexFlatL2
                M = int(hnsw_tag.replace("HNSW", ""))
            except Exception:
                M = 32

            # Create a CPU HNSW index and populate from GPU graph
            try:
                # Build an empty HNSW index
                if metric.upper() == "L2":
                    base = _faiss.IndexFlatL2(self.dimension)
                else:
                    base = _faiss.IndexFlatIP(self.dimension)
                hnsw_index = _faiss.IndexHNSWFlat(self.dimension, M)
                # The typical path would be: export graph/links from CAGRA and import to HNSW.
                # Since direct graph export APIs may vary across cuVS versions, we conservatively
                # reconstruct by adding all vectors (this is slower at build time but portable).
                hnsw_index.hnsw.efConstruction = int(self.build_params.get("efConstruction", 40))
                hnsw_index.add(xb)
            except Exception as e:
                raise RuntimeError(f"Failed to convert CAGRA to CPU HNSW (rebuild path): {e}")

            self._cpu_index = hnsw_index

    def search(self, xq: np.ndarray, topk: int):
        """Search using either GPU CAGRA or CPU HNSW depending on availability."""
        import numpy as _np
        xq = _np.asarray(xq, dtype=_np.float32)

        # Prefer CPU index if available (post-conversion)
        if self._cpu_index is not None:
            return self._cpu_index.search(xq, int(topk))

        if self._gpu_index is None:
            raise RuntimeError("CAGRA GPU index is not built. Call add() first.")

        # GPU search via cuVS
        try:
            try:
                from cuvs.neighbors import cagra as _cagra
            except Exception:
                try:
                    from cuvs.legacy.neighbors import cagra as _cagra  # type: ignore
                except Exception:
                    _cagra = None
            if _cagra is None:
                raise RuntimeError("cuVS CAGRA not found for search.")

            search_width = int(self._search_params.get("search_width", 4))
            itopk_size = int(self._search_params.get("itopk_size", max(32, topk)))
            refine_ratio = float(self._search_params.get("refine_ratio", 1.0))

            D, I = _cagra.search(
                self._gpu_index,
                xq,
                int(topk),
                search_width=search_width,
                itopk_size=itopk_size,
                refine_ratio=refine_ratio,
            )
            return D, I
        except Exception as e:
            raise RuntimeError(f"Failed to search with CAGRA: {e}")

    # Adapter-specific API to apply search-time params
    def search_with_params(self, xq: np.ndarray, topk: int, params: dict | None = None):
        self._search_params = params or {}
        return self.search(xq, topk)