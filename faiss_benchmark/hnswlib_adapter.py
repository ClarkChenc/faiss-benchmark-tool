from __future__ import annotations
import os
import numpy as np
import inspect
import time

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
        self.keep_indegree_rate = float(os.environ.get("KEEP_INDEGREE_RATE", "1.0"))

        # threads
        try:
            self._num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        except Exception:
            self._num_threads = 1

        self._index = None
        self._capacity = 0
        self._added = 0
        self._last_build_added_count = 0

    # --- Build API ---
    def init_capacity(self, max_elements: int):
        """Initialize the hnswlib index capacity and parameters."""
        if self._index is None:
            import hnswlib
            self._index = hnswlib.Index(space=self.space, dim=self.dimension)
            self._index.init_index(max_elements=int(max_elements), ef_construction=self.efConstruction, M=self.M)
            self._index.set_keep_indegree_rate(self.keep_indegree_rate)
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
        
        if self._added != self._last_build_added_count:
            if hasattr(self._index, "build_indegree_map"):
                self._index.build_indegree_map(self.keep_indegree_rate)
            self._last_build_added_count = self._added

        try:
            xq = np.asarray(xq, dtype=np.float32)
        except Exception:
            raise RuntimeError("hnswlib 适配器需要 float32 的 NumPy 数组")
        if xq.ndim != 2 or xq.shape[1] != self.dimension:
            raise RuntimeError(f"hnswlib search(): 维度不匹配，期望 {self.dimension}，得到 {xq.shape[1]}")
        labels, distances = self._index.knn_query(data=xq, k=int(topk), num_threads=self._num_threads)
        I = labels.astype(np.int64)
        D = distances.astype(np.float32)
        return D, I

    def search_with_params(self, xq: np.ndarray, topk: int, params: dict | None = None):
        if params:
            if "efSearch" in params:
                try:
                    self._index.set_ef(int(params["efSearch"]))
                except Exception:
                    pass
            if "trigger_multi_entry" in params:
                try:
                    self._index.set_trigger_multi_entry(bool(params["trigger_multi_entry"]))
                except Exception as e:
                    pass
        return self.search(xq, topk)

    def plot_histgram(self, data, output_file):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib or seaborn not found, skipping plot.")
            return

        counts = [item[1] for item in data]
        plt.figure(figsize=(8, 5))
        n, bins, patches = plt.hist(
            counts,
            bins=32,                     # 可调整：自动/指定 bin 数，或用 np.arange(min, max, step)
            range = (0, 100)
            # color='steelblue',
            # alpha=0.7,
            # edgecolor='white',
            # linewidth=1.2
        )

        # 添加细节
        plt.xlabel('Count Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Counts', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # # 可选：在柱子上标注频次
        # for i in range(len(patches)):
        #     plt.text(patches[i].get_x() + patches[i].get_width() / 2,
        #             patches[i].get_height() + max(n) * 0.01,
        #             str(int(n[i])),
        #             ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"直方图已保存为: {output_file}")

    def get_search_visit_counts(self, top_percent = 0.1, fig_save_path= "./output/stat.png"):
        """返回 (label, count) 列表，如果 hnswlib 暴露了该接口。

        需要你在本地编译的 hnswlib 中添加并暴露 C++ 方法：
        - clearSearchStat()
        - getSearchCountByLabel() -> std::vector<std::pair<labeltype, size_t>>
        """
        try:
            stat = []
            if hasattr(self._index, "getSearchCountByLabel"):
                stat = self._index.getSearchCountByLabel()
                print(f"stat: {len(stat)}")
                count = np.array([item[1] for item in stat])
                count = np.sort(count)
                print(f"count: {count}")

                sum_count = np.sum(count)
                print(f"sum_count: {sum_count}")

        except Exception:
            return
        return

    def get_in_degree_counts(self):
        """返回 (label, in_degree) 列表，如果 hnswlib 暴露了该接口。"""
        try:
            if hasattr(self._index, "getInDegreeByLabel"):
                stat = self._index.getInDegreeByLabel()
        except Exception:
            return
        return

    def get_indegree_node_hit_search_count(self, query_size):
        try:
            if hasattr(self._index, "get_hit_rate"):
                _, _, indegree_hits = self._index.get_hit_rate()
                hit_rate = float(indegree_hits) / float(query_size)
                print(f"indegree_node hit_rate: {hit_rate:.3%}")
            elif hasattr(self._index, "getHitCount"):
                hit_count = self._index.getHitCount()
                hit_rate = float(hit_count) / float(query_size)
                print(f"indegree_node hit_rate: {hit_rate:.3%}")
        except Exception:
            return
        return

    def get_stat(self, query_size):
        self.get_indegree_node_hit_search_count(query_size)

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
        inst.keep_indegree_rate = float(os.environ.get("KEEP_INDEGREE_RATE", "1.0"))
        inst._index.set_keep_indegree_rate(inst.keep_indegree_rate)
        try:
            inst._index.load_index(path, max_elements=int(max_elements))

        except Exception as e:
            raise RuntimeError(f"加载 hnswlib 索引失败: {e}")
        inst._capacity = int(max_elements)
        inst._added = int(max_elements)  # assume fully populated
        inst._last_build_added_count = int(max_elements)
        inst._num_threads = int(num_threads) if (num_threads is not None) else int(os.environ.get("OMP_NUM_THREADS", "1"))
        

        try:
            inst._index.set_num_threads(inst._num_threads)
        except Exception as e:
            pass
        return inst

class HnswlibSplitIndexAdapter:
    def __init__(self, dimension: int, build_params: dict | None = None):
        bp = build_params or {}
        self.dimension = int(dimension)
        self.space = str(bp.get("space", "l2"))
        self.M = int(bp.get("M", 16))
        self.efConstruction = int(bp.get("efConstruction", 200))
        if "keep_indegree_rate" in bp:
             self.keep_indegree_rate = float(bp["keep_indegree_rate"])
        else:
            try:
                self.keep_indegree_rate = float(os.environ.get("KEEP_INDEGREE_RATE", "1.0"))
            except Exception:
                self.keep_indegree_rate = 1.0
                
        self.seg_num = int(bp.get("seg_num", 1))
        self.is_merge = bool(bp.get("is_merge", False))
        self.merge_ratio = float(bp.get("merge_ratio", 1.0))
        try:
            self._num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        except Exception:
            self._num_threads = 1
            
        self._segments = []
        self._capacity_total = 0
        self._added_total = 0
        self._merged_index = None
        self._merged_index_map_built = False
        self.segment_sizes = None

        class _HnswParamsProxy:
            def __init__(self, owner):
                self._owner = owner
                self._efSearch = None
            @property
            def efSearch(self):
                return self._efSearch
            @efSearch.setter
            def efSearch(self, v):
                self._efSearch = int(v)
                if self._owner._merged_index:
                    try:
                        self._owner._merged_index.set_ef(int(v))
                    except Exception:
                        pass
                
                if self._owner._segments:
                    for seg in self._owner._segments:
                        try:
                            seg["index"].set_ef(int(v))
                        except Exception:
                            pass
        self.hnsw = _HnswParamsProxy(self)

    def init_capacity(self, max_elements: int):
        if self._segments:
            return
        import hnswlib
        max_elements = int(max_elements)
        self._capacity_total = max_elements
        seg_cap = (max_elements + self.seg_num - 1) // self.seg_num
        for _ in range(self.seg_num):
            idx = hnswlib.Index(space=self.space, dim=self.dimension)
            idx.init_index(max_elements=int(seg_cap), ef_construction=self.efConstruction, M=self.M)
            idx.set_keep_indegree_rate(self.keep_indegree_rate)
            try:
                idx.set_num_threads(self._num_threads)
            except Exception:
                pass
            self._segments.append({"index": idx, "capacity": int(seg_cap), "added": 0})

    def train(self, xb: np.ndarray):
        return None

    def add(self, xb: np.ndarray):
        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dimension:
            raise RuntimeError(f"hnswlib add(): 维度不匹配，期望 {self.dimension}，得到 {xb.shape[1]}")
        if not self._segments:
            self.init_capacity(int(xb.shape[0]))
        if self._added_total + xb.shape[0] > self._capacity_total:
            raise RuntimeError(f"hnswlib 索引容量不足: 当前总容量 {self._capacity_total}, 试图添加 {self._added_total + xb.shape[0]}")
        labels = np.arange(self._added_total, self._added_total + xb.shape[0], dtype=np.int64)
        start = 0
        remain = xb.shape[0]
        for seg in self._segments:
            if remain <= 0:
                break
            can_add = min(remain, seg["capacity"] - seg["added"])
            if can_add <= 0:
                continue
            xb_slice = xb[start:start+can_add]
            lab_slice = labels[start:start+can_add]
            seg["index"].add_items(xb_slice, lab_slice)
            seg["added"] += int(can_add)
            start += can_add
            remain -= can_add
        self._added_total += xb.shape[0]

    def search(self, xq: np.ndarray, topk: int):
        if self._merged_index:
            if not self._merged_index_map_built:
                if hasattr(self._merged_index, "build_indegree_map"):
                    self._merged_index.build_indegree_map(self.keep_indegree_rate)
                self._merged_index_map_built = True

            xq = np.asarray(xq, dtype=np.float32)
            if xq.ndim != 2 or xq.shape[1] != self.dimension:
                raise RuntimeError(f"hnswlib search(): 维度不匹配，期望 {self.dimension}，得到 {xq.shape[1]}")
            labels, distances = self._merged_index.knn_query(data=xq, k=int(topk), num_threads=self._num_threads)
            D = distances.astype(np.float32)
            I = labels.astype(np.int64)
            return D, I

        if not self._segments:
            raise RuntimeError("hnswlib search(): 索引尚未构建")
        xq = np.asarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dimension:
            raise RuntimeError(f"hnswlib search(): 维度不匹配，期望 {self.dimension}，得到 {xq.shape[1]}")
        all_D = []
        all_I = []
        for seg in self._segments:
            labels, distances = seg["index"].knn_query(data=xq, k=int(topk), num_threads=self._num_threads)
            all_I.append(labels.astype(np.int64))
            all_D.append(distances.astype(np.float32))
        Dcat = np.concatenate(all_D, axis=1) if len(all_D) > 1 else all_D[0]
        Icat = np.concatenate(all_I, axis=1) if len(all_I) > 1 else all_I[0]
        k = int(topk)
        if Dcat.shape[1] <= k:
            order = np.argsort(Dcat, axis=1)
            rows = np.arange(Dcat.shape[0])[:, None]
            Dk = Dcat[rows, order[:, :k]]
            Ik = Icat[rows, order[:, :k]]
            return Dk, Ik
        idx = np.argpartition(Dcat, kth=k-1, axis=1)[:, :k]
        rows = np.arange(Dcat.shape[0])[:, None]
        Dk = Dcat[rows, idx]
        Ik = Icat[rows, idx]
        ord2 = np.argsort(Dk, axis=1)
        Dk_sorted = Dk[rows, ord2]
        Ik_sorted = Ik[rows, ord2]
        return Dk_sorted, Ik_sorted

    def search_with_params(self, xq: np.ndarray, topk: int, params: dict | None = None):
        if params:
            if"efSearch" in params:
                try:
                    self.hnsw.efSearch = int(params["efSearch"])
                except Exception:
                    pass
            if "trigger_multi_entry" in params:
                try:
                    self._merged_index.set_trigger_multi_entry(bool(params["trigger_multi_entry"]))
                except Exception as e:
                    print(f"Failed to set trigger_multi_entry: {e}")
                    pass

        return self.search(xq, topk)

    def get_indegree_node_hit_search_count(self, query_size):
        try:
            if hasattr(self._merged_index, "get_hit_rate"):
                _, _, indegree_hits = self._merged_index.get_hit_rate()
                hit_rate = float(indegree_hits) / float(query_size)
                print(f"indegree_node hit_rate: {hit_rate:.3%}")
            elif hasattr(self._merged_index, "getHitCount"):
                hit_count = self._merged_index.getHitCount()
                hit_rate = float(hit_count) / float(query_size)
                print(f"indegree_node hit_rate: {hit_rate:.3%}")
        except Exception:
            return
        return

    def get_stat(self, query_size):
        self.get_indegree_node_hit_search_count(query_size)

    def save_to_cache(self, path: str):
        os.makedirs(path, exist_ok=True)
        seg_paths = []
        for i, seg in enumerate(self._segments):
            fp = os.path.join(path, f"segment_{i}.hnswlib")
            seg["index"].save_index(fp)
            seg_paths.append(fp)
        
        if self.is_merge:
            import hnswlib
            merged_path = os.path.join(path, "merged_index.hnswlib")
            try:
                start_time = time.time()
                # Merge indices
                merged_idx = hnswlib.merge_indices(
                    filenames=seg_paths, 
                    space_name=self.space, 
                    dim=self.dimension, 
                    total_max_elements=self._capacity_total, 
                    M=self.M, 
                    ef_construction=self.efConstruction,
                    random_seed=100,
                    ratio=self.merge_ratio,
                    keep_pruned_connections=self.keep_indegree_rate
                )
                merge_time = time.time() - start_time
                print(f"merge index use time: {merge_time:.3f}s")

                merged_idx.save_index(merged_path)
                self._merged_index = merged_idx
                if self.segment_sizes is None and self._segments:
                    self.segment_sizes = [seg["added"] for seg in self._segments]
                if self.segment_sizes and hasattr(self._merged_index, "set_segment_boundaries"):
                     ends = np.cumsum(self.segment_sizes, dtype=np.uint64)
                     self._merged_index.set_segment_boundaries(ends)
            except Exception as e:
                print(f"Warning: Failed to merge indices: {e}")

    @classmethod
    def load_from_cache(cls, dir_path: str, dimension: int, space: str, seg_num: int, segment_sizes: list[int], num_threads: int | None = None, is_merge: bool = False):
        import hnswlib
        inst = cls(dimension=dimension, build_params={"space": space, "seg_num": int(seg_num), "is_merge": is_merge})
        
        merged_path = os.path.join(dir_path, "merged_index.hnswlib")
        if is_merge and os.path.exists(merged_path):
            inst._capacity_total = int(sum(segment_sizes))
            inst._added_total = int(sum(segment_sizes))
            
            idx = hnswlib.Index(space=space, dim=int(dimension))
            idx.set_keep_indegree_rate(inst.keep_indegree_rate)
            idx.load_index(merged_path, max_elements=inst._capacity_total)
            try:
                idx.set_num_threads(int(num_threads) if (num_threads is not None) else int(os.environ.get("OMP_NUM_THREADS", "1")))
            except Exception:
                pass
            inst._merged_index = idx
            inst.segment_sizes = list(segment_sizes)
            if inst.segment_sizes and hasattr(inst._merged_index, "set_segment_boundaries"):
                ends = np.cumsum(inst.segment_sizes, dtype=np.uint64)
                inst._merged_index.set_segment_boundaries(ends)
            return inst
        
        inst._segments = []
        inst._capacity_total = int(sum(segment_sizes))
        inst._added_total = int(sum(segment_sizes))
        for i in range(int(seg_num)):
            fp = os.path.join(dir_path, f"segment_{i}.hnswlib")
            cap = int(segment_sizes[i]) if i < len(segment_sizes) else 0
            idx = hnswlib.Index(space=space, dim=int(dimension))
            idx.set_keep_indegree_rate(inst.keep_indegree_rate)
            idx.load_index(fp, max_elements=cap)
            try:
                idx.set_num_threads(int(num_threads) if (num_threads is not None) else int(os.environ.get("OMP_NUM_THREADS", "1")))
            except Exception:
                pass
            inst._segments.append({"index": idx, "capacity": cap, "added": cap})
        return inst

    def get_cumulative_hit_rate(self):
        if self._merged_index and hasattr(self._merged_index, "get_hit_rate"):
            return self._merged_index.get_hit_rate()
        return (0, 0)
