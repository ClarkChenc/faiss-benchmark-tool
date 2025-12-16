import numpy as np
import os
import scann
import shutil

class ScannIndexAdapter:
    """
    ScaNN 适配器，提供与 Faiss 相似的接口：train/add/search/search_with_params。

    设计要点：
    - 构建参数在 finalize_build 阶段应用，add() 仅收集基向量。
    - 支持两种距离：dot_product 和 squared_l2。
    - 支持可选构建项：tree(num_leaves, leaf_size)、AH(score_ah)、reorder(reorder_k)。
    - 搜索时使用 search_batched，返回 (D, I) 与 Faiss 格式一致。

    限制：
    - ScaNN 不支持真正的增量 add；本适配器会缓冲新增并在 finalize_build 时一次性构建。
    - 如果未调用 finalize_build，首次 search 会自动触发构建。
    """

    def __init__(self, dimension: int, build_params: dict | None = None):
        self.dimension = int(dimension)
        self.build_params = build_params or {}

        # Internal handles
        self._base = None  # np.ndarray[float32]
        self._searcher = None  # scann.searcher

        # Last applied search params
        self._search_params = {}

        # Build timing cache (filled in finalize_build or lazy build on search)
        self._train_time = 0.0
        self._add_time = 0.0
        self._count = 0

        self._num_threads = int(os.environ["OMP_NUM_THREADS"])

    # Faiss-like interface
    def train(self, xb: np.ndarray):
        """ScaNN 无需训练，保持接口兼容。"""
        return None

    def add(self, xb: np.ndarray):
        """缓冲基向量，延迟到 finalize_build 统一构建 ScaNN 索引。"""
        try:
            xb = np.asarray(xb, dtype=np.float32)
        except Exception:
            raise RuntimeError("ScaNN 适配器需要 float32 的 NumPy 数组")

        if xb.ndim != 2 or xb.shape[1] != self.dimension:
            raise RuntimeError(f"ScaNN add(): 维度不匹配，期望 {self.dimension}，得到 {xb.shape[1]}")

        if self._base is None:
            self._base = xb.copy()
        else:
            # 累积基向量
            self._base = np.concatenate([self._base, xb], axis=0)
        print("add vectors, now shape is: ", self._base.shape)


    def _build_searcher_v3(self):
        builder = scann.scann_ops_pybind.builder(self._base, 50, 'squared_l2')
        self._searcher = builder.tree(10000, 100).score_ah(2, 0.1).reorder(800).set_n_training_threads(self._num_threads).build()
        if self._searcher is None:
            raise(f"get none searcher")
        self._searcher.set_num_threads(self._num_threads)


        #self._searcher = builder.autopilot().build()
        print("config: ", builder.create_config())

    def _build_searcher_v4(self):        
        num_neighbors = int(self.build_params.get("num_neighbors", 10))
        num_leaves = int(self.build_params.get("num_leaves"))
        num_leaves_to_search = int(self.build_params.get("num_leaves_to_search", num_leaves))
        if num_leaves is None or num_leaves_to_search is None:
            raise RuntimeError(f"index_param.num_leaves or index_param.num_leaves_to_search not set")

        ah_bits = int(self.build_params.get("ah_bits"))
        ah_threshold = float(self.build_params.get("ah_threshold"))
        if ah_bits is None or ah_threshold is None:
            raise RuntimeError(f"index_param.ah_bits or index_param.ah_threshold not set")

        reorder_k = int(self.build_params.get("reorder_k"))
        if reorder_k is None:
            raise RuntimeError(f"index_param.reorder_k not set")
        
        builder = scann.scann_ops_pybind.builder(self._base, num_neighbors, "squared_l2")
        train_data_size = int(self._base.shape[0] * 0.1)
        max_train_data_size = 1000000
        if train_data_size > max_train_data_size:
            train_data_size = max_train_data_size
        
        if num_leaves < 0 :
            self._searcher = builder.autopilot().build()
        else:
            self._searcher = (builder.tree(num_leaves, num_leaves_to_search, training_sample_size=train_data_size)
                            .score_ah(ah_bits, ah_threshold).reorder(reorder_k).set_n_training_threads(self._num_threads)
                            .build())
                        
        if self._searcher is None:
            raise RuntimeError(f"searcher init failed")
        self._searcher.set_num_threads(self._num_threads)
        print ("set num threads", self._num_threads)
        print(f"builder config: {builder.create_config()}")

    def finalize_build(self):
        """执行实际构建并返回构建时间信息。"""
        import time
        t0 = time.time()
        self._build_searcher_v4()
        self._add_time = time.time() - t0
        self._base = None
        return {
            "train_time": self._train_time,
            "add_time": self._add_time,
            "gpu_mem_peak_used_bytes": None,
            "gpu_mem_total_bytes": None,
        }

    def search(self, xq: np.ndarray, topk: int):
        # ScaNN 使用 search_batched，支持 final_num_neighbors
        final_k = int(self._search_params.get("final_num_neighbors", topk))
        # 保证最终返回 topk 形状
        request_k = max(final_k, int(topk))
        if self._searcher is None:
            print(f"_searcher is none ")

        neighbors, distances = self._searcher.search_batched_parallel(xq, request_k)

        # 截断或填充到 topk 形状
        if neighbors.shape[1] != topk:
            # 截断多余列
            if neighbors.shape[1] > topk:
                neighbors = neighbors[:, :topk]
                distances = distances[:, :topk]
            else:
                # 填充不足列
                pad = topk - neighbors.shape[1]
                neg_inf = _np.full((neighbors.shape[0], pad), _np.inf, dtype=distances.dtype)
                minus_one = _np.full((neighbors.shape[0], pad), -1, dtype=neighbors.dtype)
                distances = _np.concatenate([distances, neg_inf], axis=1)
                neighbors = _np.concatenate([neighbors, minus_one], axis=1)

        # 统一返回格式 (D, I) 与 Faiss 对齐
        I = neighbors.astype(np.int64)
        D = distances.astype(np.float32)
        return D, I

    def search_with_params(self, xq: np.ndarray, topk: int, params: dict | None = None):
        self._search_params = params or {}
        return self.search(xq, topk)

    # --- Cache/Serialization helpers ---
    def save_to_cache(self, dir_path: str):
        """Save the built ScaNN searcher to the given path."""
        if self._searcher is None:
            raise RuntimeError("ScaNN searcher is not built; finalize_build() required before saving")
        try:
            # Prefer pybind save API when available
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            from importlib.metadata import version
            if version("scann") == "1.4.2":
                self._searcher.serialize(dir_path, relative_path=True)
            else:
                self._searcher.serialize(dir_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save ScaNN searcher: {e}")

    @classmethod
    def load_from_cache(cls, dir_path: str, num_threads: int | None = None):
        """Load a ScaNN searcher from cache path and wrap into adapter.

        Dimension is not required for search; we set a placeholder.
        """
        try:
            searcher = scann.scann_ops_pybind.load_searcher(dir_path)
        except Exception as e:
            print(f"Failed to load ScaNN searcher: {e}")
            raise RuntimeError(f"Failed to load ScaNN searcher: {e}")

        inst = cls(dimension=0, build_params={})
        inst._searcher = searcher
        inst._num_threads = int(num_threads) if (num_threads is not None) else int(os.environ.get("OMP_NUM_THREADS", "1"))
        try:
            inst._searcher.set_num_threads(inst._num_threads)
        except Exception:
            pass
        return inst
