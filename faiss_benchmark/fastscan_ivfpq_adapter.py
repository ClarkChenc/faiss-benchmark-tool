import os
import re
import numpy as np
import faiss


class FastScanIVFPQAdapter:
    """Adapter using faiss.IndexIVFPQFastScan to build/search FastScan IVF-PQ indexes (CPU).

    Build params (index_param in config.yaml):
    - metric: 'L2' or 'IP' (default 'L2')
    - ivf_nlist or nlist: number of coarse clusters (default 1024)
    - pq_m: number of PQ sub-quantizers (default 32)
    - pq_bits: bits per sub-quantizer (default 4 for FastScan)
    - fs_residual: whether to use residual FastScan (default False)
    - refine: optional 'RFlat' to enable exact re-ranking with configurable k_factor

    Search params (search_param in config.yaml):
    - nprobe: clusters to probe
    - k_factor: re-ranking width when refine='RFlat' (topk*k_factor candidates)
    """

    def __init__(self, index_type: str, dimension: int, build_params: dict | None = None):
        self.index_type = str(index_type)
        self.dimension = int(dimension)
        self.build_params = build_params or {}
        self.metric = faiss.METRIC_L2

        # Parse nlist either from index_type (e.g., "IVF1024,IVFPQFastScan") or from params
        nlist = int(self.build_params.get("nlist", 1024))

        pq_m = int(self.build_params.get("pq_m", 32))
        pq_bits = int(self.build_params.get("pq_bits", 4))
        fs_residual = bool(self.build_params.get("fs_residual", False))

        # Coarse quantizer according to metric
        quantizer = faiss.IndexFlatL2(self.dimension) if self.metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)

        # Construct IndexIVFPQFastScan
        self._index = faiss.IndexIVFPQFastScan(quantizer, self.dimension, nlist, pq_m, pq_bits, self.metric)

        # Optional refine using exact Flat re-ranking
        refine_opt = self.build_params.get("refine")
        if isinstance(refine_opt, str) and refine_opt.strip().upper() == "RFLAT":
            try:
                self._index = faiss.IndexRefineFlat(self._index)
            except Exception:
                # If IndexRefineFlat not available, silently ignore refine
                pass

    # --- Build API ---
    def train(self, xb: np.ndarray):
        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dimension:
            raise RuntimeError(f"FastScanIVFPQ train(): 维度不匹配，期望 {self.dimension}，得到 {xb.shape[1]}")
        # IVFPQFastScan requires training
        self._index.train(xb)

    def add(self, xb: np.ndarray):
        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dimension:
            raise RuntimeError(f"FastScanIVFPQ add(): 维度不匹配，期望 {self.dimension}，得到 {xb.shape[1]}")
        self._index.add(xb)

    # --- Search API ---
    def search(self, xq: np.ndarray, topk: int):
        xq = np.asarray(xq, dtype=np.float32)
        D, I = self._index.search(xq, int(topk))
        return D.astype(np.float32), I.astype(np.int64)

    def search_with_params(self, xq: np.ndarray, topk: int, params: dict | None = None):
        p = params or {}

        base_params = faiss.SearchParametersIVF()
        base_params.nprobe = int(p["nprobe"])

        refine_params = faiss.IndexRefineSearchParameters(k_factor=p["k_factor"])
        refine_params.base_index_params = base_params


        xq = np.asarray(xq, dtype=np.float32)
        D, I = self._index.search(xq, int(topk), params=refine_params)
        return D.astype(np.float32), I.astype(np.int64)

    # --- Cache helpers ---
    def save_to_cache(self, path: str):
        try:
            faiss.write_index(self._index, path)
        except Exception as e:
            raise RuntimeError(f"保存 FastScanIVFPQ 索引失败: {e}")

    def get_cpu_index(self):
        """Expose underlying Faiss CPU index for caching/routing."""
        return self._index

    @classmethod
    def load_from_cache(cls, path: str, dimension: int, build_params: dict | None = None):
        try:
            idx = faiss.read_index(path)
        except Exception as e:
            raise RuntimeError(f"加载 FastScanIVFPQ 索引失败: {e}")
        inst = cls(index_type="IVFPQFastScan", dimension=int(dimension), build_params=build_params or {})
        inst._index = idx

        return inst
