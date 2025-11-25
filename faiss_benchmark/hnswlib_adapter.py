import os
import numpy as np


class HnswlibIndexAdapter:
    """
    Adapter that exposes a Faiss-like interface on top of hnswlib (CPU-only).

    Build params:
    - space: 'l2' or 'cosine' (default: 'l2')
    - M: graph degree (default: 16)
    - efConstruction: construction breadth (default: 200)

    Search params:
    - efSearch: search breadth

    Methods implemented: train (no-op), add, search, search_with_params,
    save_to_cache, load_from_cache, init_capacity.
    """

    def __init__(self, dimension: int, build_params: dict | None = None):
        bp = build_params or {}
        self.dimension = int(dimension)
        self.space = str(bp.get("space", "l2"))
        self.M = int(bp.get("M", 16))
        self.efConstruction = int(bp.get("efConstruction", 200))
        # threads
        try:
            self._num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        except Exception:
            self._num_threads = 1

        self._index = None
        self._capacity = 0
        self._added = 0

    # --- Build API ---
    def init_capacity(self, max_elements: int):
        """Initialize the hnswlib index capacity and parameters."""
        if self._index is None:
            import hnswlib
            self._index = hnswlib.Index(space=self.space, dim=self.dimension)
            self._index.init_index(max_elements=int(max_elements), ef_construction=self.efConstruction, M=self.M)
            try:
                self._index.set_num_threads(self._num_threads)
            except Exception:
                pass
            self._capacity = int(max_elements)

    def train(self, xb: np.ndarray):
        # hnswlib does not require training
        return None

    def add(self, xb: np.ndarray):
        try:
            xb = np.asarray(xb, dtype=np.float32)
        except Exception:
            raise RuntimeError("hnswlib 适配器需要 float32 的 NumPy 数组")
        if xb.ndim != 2 or xb.shape[1] != self.dimension:
            raise RuntimeError(f"hnswlib add(): 维度不匹配，期望 {self.dimension}，得到 {xb.shape[1]}")
        if self._index is None:
            # Fallback: initialize with current batch size; benchmark will pre-set full capacity
            self.init_capacity(int(xb.shape[0]))
        # Ensure capacity is sufficient
        if self._added + xb.shape[0] > self._capacity:
            raise RuntimeError(
                f"hnswlib 索引容量不足: 当前容量 {self._capacity}, 试图添加 {self._added + xb.shape[0]}"
            )
        self._index.add_items(xb)
        self._added += int(xb.shape[0])

    # --- Search API ---
    def search(self, xq: np.ndarray, topk: int):
        if self._index is None:
            raise RuntimeError("hnswlib search(): 索引尚未构建")
        try:
            xq = np.asarray(xq, dtype=np.float32)
        except Exception:
            raise RuntimeError("hnswlib 适配器需要 float32 的 NumPy 数组")
        if xq.ndim != 2 or xq.shape[1] != self.dimension:
            raise RuntimeError(f"hnswlib search(): 维度不匹配，期望 {self.dimension}，得到 {xq.shape[1]}")
        labels, distances = self._index.knn_query(xq, k=int(topk), num_threads=self._num_threads)
        I = labels.astype(np.int64)
        D = distances.astype(np.float32)
        return D, I

    def search_with_params(self, xq: np.ndarray, topk: int, params: dict | None = None):
        if params and "efSearch" in params:
            try:
                self._index.set_ef(int(params["efSearch"]))
            except Exception:
                pass
        return self.search(xq, topk)

    # --- Cache/Serialization helpers ---
    def save_to_cache(self, path: str):
        if self._index is None:
            raise RuntimeError("hnswlib 索引尚未构建，无法保存")
        try:
            self._index.save_index(path)
        except Exception as e:
            raise RuntimeError(f"保存 hnswlib 索引失败: {e}")

    @classmethod
    def load_from_cache(cls, path: str, dimension: int, space: str, max_elements: int, num_threads: int | None = None):
        import hnswlib
        inst = cls(dimension=dimension, build_params={"space": space})
        inst._index = hnswlib.Index(space=space, dim=int(dimension))
        try:
            inst._index.load_index(path, max_elements=int(max_elements))
        except Exception as e:
            raise RuntimeError(f"加载 hnswlib 索引失败: {e}")
        inst._capacity = int(max_elements)
        inst._added = int(max_elements)  # assume fully populated
        inst._num_threads = int(num_threads) if (num_threads is not None) else int(os.environ.get("OMP_NUM_THREADS", "1"))

        try:
            inst._index.set_num_threads(inst._num_threads)
        except Exception:
            pass
        return inst

