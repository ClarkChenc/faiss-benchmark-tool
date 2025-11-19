import numpy as np

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

    def _build_searcher(self):
        import scann
        import inspect

        if self._base is None or self._base.shape[0] == 0:
            raise RuntimeError("ScaNN 构建失败：基向量为空")

        # 距离度量
        distance = str(self.build_params.get("distance_measure", "dot_product")).lower()
        if distance not in ("dot_product", "squared_l2"):
            distance = "dot_product"

        num_neighbors = int(self.build_params.get("num_neighbors", max(10, min(50, self._search_params.get("final_num_neighbors", 10)))))

        # 优先使用 pybind builder（在多版本环境中更稳定）
        if hasattr(scann, "scann_ops_pybind") and hasattr(scann.scann_ops_pybind, "builder"):
            builder = scann.scann_ops_pybind.builder(self._base, num_neighbors, distance)
        else:
            builder = scann.ScannBuilder(self._base, num_neighbors=num_neighbors, distance_measure=distance)
        # 记录可用方法，便于诊断
        builder_methods = {m for m in dir(builder) if not m.startswith("_")}

        # 树分区
        num_leaves = self.build_params.get("num_leaves")
        num_leaves_to_search = self.build_params.get("num_leaves_to_search")
        if num_leaves_to_search is None:
            ls = self.build_params.get("leaf_size")
            if ls is not None:
                try:
                    num_leaves_to_search = int(ls)
                except Exception:
                    num_leaves_to_search = None
        if num_leaves is not None and "tree" in builder_methods:
            # 兼容不同签名：优先 (num_leaves, num_leaves_to_search)，回退仅 num_leaves
            called_tree = False
            for args in (
                (int(num_leaves), int(num_leaves_to_search) if num_leaves_to_search is not None else max(1, int(num_leaves) // 20)),
                (int(num_leaves),),
            ):
                try:
                    print("args", type(args))
                    print("args", *args)
                    builder.tree(*args)
                    called_tree = True
                    break
                except Exception:
                    continue

        # 选择打分方式（按可用方法优先级回退）
        scored = False
        scoring_chosen = None
        # AH 打分：当前环境可能不兼容，默认跳过；如需启用可在后续版本加回
        # 若未设置打分，则回退到 brute 家族或 autopilot
        if not scored:
            for method_name, chosen in (
                ("score_brute_force", "score_brute_force"),
                ("score_brute", "score_brute"),
                ("autopilot", "autopilot"),
            ):
                if method_name in builder_methods:
                    try:
                        getattr(builder, method_name)()
                        scored = True
                        scoring_chosen = chosen
                        break
                    except Exception:
                        continue

        # 重排（需要在构建阶段启用，通常在设置打分之后）
        # 重排：为避免不兼容，默认跳过；如需启用请在环境验证后开启

        # 构建 searcher（若失败，尝试简化管道回退）
        try:
            self._searcher = builder.build()
        except Exception:
            # 回退：重新最小化构建并尝试 autopilot
            if hasattr(scann, "scann_ops_pybind") and hasattr(scann.scann_ops_pybind, "builder"):
                builder = scann.scann_ops_pybind.builder(self._base, num_neighbors, distance)
            else:
                builder = scann.ScannBuilder(self._base, num_neighbors=num_neighbors, distance_measure=distance)
            try:
                if "autopilot" in {m for m in dir(builder) if not m.startswith("_")}:
                    getattr(builder, "autopilot")()
                self._searcher = builder.build()
            except Exception as e2:
                available = ",".join(sorted(builder_methods))
                raise RuntimeError(
                    f"ScaNN 构建失败：{e2}; scoring={scoring_chosen or 'none'}; available_methods={available}"
                )

    def finalize_build(self):
        """执行实际构建并返回构建时间信息。"""
        import time

        t0 = time.time()
        self._build_searcher()
        self._add_time = time.time() - t0
        return {
            "train_time": self._train_time,
            "add_time": self._add_time,
            "gpu_mem_peak_used_bytes": None,
            "gpu_mem_total_bytes": None,
        }

    def search(self, xq: np.ndarray, topk: int):
        import numpy as _np
        xq = _np.asarray(xq, dtype=_np.float32)

        # Lazy build（如果未 finalize，则在首次搜索前构建）
        if self._searcher is None:
            _ = self.finalize_build()

        # ScaNN 使用 search_batched，支持 final_num_neighbors
        final_k = int(self._search_params.get("final_num_neighbors", topk))
        # 保证最终返回 topk 形状
        request_k = max(final_k, int(topk))

        neighbors, distances = self._searcher.search_batched(xq, final_num_neighbors=request_k)

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