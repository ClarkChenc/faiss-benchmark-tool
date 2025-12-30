#!/usr/bin/env python3
import argparse
import os
import sys
import json
import glob
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def _find_meta_path(index_path: str) -> str | None:
    p = os.path.abspath(index_path)
    if os.path.isdir(p):
        base_dir = p
        parent = os.path.dirname(base_dir)
        base_name = os.path.basename(base_dir)
        if base_name.endswith(".hnswlib"):
            meta = os.path.join(parent, base_name[:-8] + "_meta.json")
            if os.path.exists(meta):
                return meta
        # fallback: pick the only *_meta.json in parent
        candidates = glob.glob(os.path.join(parent, "*_meta.json"))
        if candidates:
            return candidates[0]
        return None
    else:
        # file path: try parent directory of the index dir
        base_dir = os.path.dirname(p)
        parent = os.path.dirname(base_dir)
        base_name = os.path.basename(base_dir)
        if base_name.endswith(".hnswlib"):
            meta = os.path.join(parent, base_name[:-8] + "_meta.json")
            if os.path.exists(meta):
                return meta
        # fallback: search parent for *_meta.json
        candidates = glob.glob(os.path.join(parent, "*_meta.json"))
        if candidates:
            return candidates[0]
        return None

def _resolve_index_file(index_path: str) -> str:
    p = os.path.abspath(index_path)
    if os.path.isdir(p):
        merged = os.path.join(p, "merged_index.hnswlib")
        if os.path.exists(merged):
            return merged
        # fallback to the first segment file
        segs = sorted(glob.glob(os.path.join(p, "segment_*.hnswlib")))
        if segs:
            return segs[0]
        raise FileNotFoundError(f"没有找到 merged_index.hnswlib 或 segment_*.hnswlib 在目录 {p}")
    else:
        return p

def load_index(index_path, space, dim):
    import hnswlib
    idx = hnswlib.Index(space=space, dim=int(dim))
    # max_elements=0 让库自行解析
    idx.load_index(index_path, max_elements=0)
    return idx

def parse_l0_neighbors(idx, start_id, end_id, output_json=False):
    params = idx.get_index_params()
    offset_l0 = int(params["offset_level0"])
    size_per = int(params["size_data_per_element"])
    cur_count = int(params["cur_element_count"])
    ext = np.array(params["label_lookup_external"], dtype=np.int64)
    itn = np.array(params["label_lookup_internal"], dtype=np.uint32)
    inv = {int(itn[i]): int(ext[i]) for i in range(len(ext))}
    data = np.array(params["data_level0"], dtype=np.uint8)

    res = {}
    for doc_id in range(int(start_id), int(end_id)):
        if doc_id not in set(ext.tolist()):
            continue
        try:
            internal = int(itn[np.where(ext == doc_id)[0][0]])
        except Exception:
            continue
        if internal < 0 or internal >= cur_count:
            continue
        base = internal * size_per + offset_l0
        deg = int(np.frombuffer(data[base:base+4].tobytes(), dtype=np.int32)[0])
        nb_start = base + 4
        nbs = np.frombuffer(data[nb_start:nb_start + 4 * deg].tobytes(), dtype=np.uint32)
        res[doc_id] = [inv.get(int(x), -1) for x in nbs.tolist()]

    if output_json:
        import json
        print(json.dumps(res, ensure_ascii=False))
    else:
        for k in sorted(res.keys()):
            print(f"{k}\t" + ",".join(str(x) for x in res[k]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index_cache 生成的索引路径（目录或文件）")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    meta_path = _find_meta_path(args.index)
    if not meta_path or not os.path.exists(meta_path):
        raise RuntimeError("未找到缓存的 meta.json。请传入 index_cache 保存的路径（包含 *_meta.json）。")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f) or {}
    space = str(meta.get("space", "l2"))
    dim = int(meta.get("dimension", 0))
    if dim <= 0:
        raise RuntimeError("meta.json 中缺少有效的 dimension 信息。")

    target_index = _resolve_index_file(args.index)
    idx = load_index(target_index, space, dim)
    parse_l0_neighbors(idx, args.start, args.end, output_json=args.json)

if __name__ == "__main__":
    main()
