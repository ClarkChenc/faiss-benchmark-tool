#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

def detect_format(path: str, explicit: str | None) -> str:
    if explicit:
        fmt = explicit.lower()
        if fmt not in ("fvecs", "ivecs"):
            raise ValueError(f"Unsupported format '{explicit}'. Use 'fvecs' or 'ivecs'.")
        return fmt
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fvecs", ".fvec"):
        return "fvecs"
    if ext in (".ivecs", ".ivec"):
        return "ivecs"
    raise ValueError(f"Cannot infer format from extension '{ext}'. Please pass --format.")

def read_header(path: str) -> tuple[int, int, int]:
    """
    Returns (dimension, num_vectors, record_bytes).
    Assumes constant dimension across records, TexMex layout:
      [int32 d][d * (float32 or int32)]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        # Read first int32 as dimension
        head = np.fromfile(f, dtype=np.int32, count=1)
        if head.size != 1:
            raise ValueError("File too small: cannot read dimension header")
        d = int(head[0])
    # record size is 4 * (d + 1) bytes (1 int32 header + d values)
    record_bytes = 4 * (d + 1)
    if record_bytes <= 0:
        raise ValueError(f"Invalid record_bytes computed: {record_bytes}")
    if file_size % record_bytes != 0:
        # Warn but still compute floor division
        print(f"Warning: file_size {file_size} not multiple of record_bytes {record_bytes}", file=sys.stderr)
    n = file_size // record_bytes
    return d, n, record_bytes

def read_slice(path: str, start: int, count: int, fmt: str) -> np.ndarray:
    """Read a slice [start:start+count] of vectors without loading the entire file."""
    d, n, record_bytes = read_header(path)
    if start < 0 or start >= n:
        raise IndexError(f"start {start} out of range [0, {n})")
    if count <= 0:
        raise ValueError("count must be > 0")
    end = min(start + count, n)
    num = end - start
    with open(path, "rb") as f:
        # Seek to the start record
        f.seek(start * record_bytes, os.SEEK_SET)
        # Read num records worth of int32s: each record has (d+1) int32s
        raw = np.fromfile(f, dtype=np.int32, count=num * (d + 1))
        if raw.size != num * (d + 1):
            raise ValueError(f"Unexpected read size: got {raw.size}, expected {num * (d + 1)}")
        arr = raw.reshape(num, d + 1)
        if fmt == "fvecs":
            data = arr[:, 1:].view(np.float32).reshape(num, d)
        elif fmt == "ivecs":
            data = arr[:, 1:]
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        return data

def print_summary(path: str, fmt: str):
    d, n, _ = read_header(path)
    print(f"File: {path}")
    print(f"Format: {fmt}")
    print(f"Dimension: {d}")
    print(f"Vectors: {n}")

def main():
    parser = argparse.ArgumentParser(
        description="View TexMex .fvecs/.ivecs files: show header and slices."
    )
    parser.add_argument("--file", required=True, help="Path to .fvecs or .ivecs file")
    parser.add_argument("--format", choices=["fvecs", "ivecs"], help="Explicit file format if extension is ambiguous")
    parser.add_argument("--summary", action="store_true", help="Print header summary only")
    parser.add_argument("--start", type=int, default=0, help="Start vector index (default: 0)")
    parser.add_argument("--count", type=int, default=5, help="Number of vectors to show (default: 5)")
    parser.add_argument("--stats", action="store_true", help="Print basic statistics for the slice")
    parser.add_argument("--precision", type=int, default=4, help="Float precision when printing fvecs")
    args = parser.parse_args()

    try:
        fmt = detect_format(args.file, args.format)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        print_summary(args.file, fmt)
    except Exception as e:
        print(f"Error reading header: {e}", file=sys.stderr)
        sys.exit(2)

    if args.summary:
        return

    try:
        data = read_slice(args.file, args.start, args.count, fmt)
    except Exception as e:
        print(f"Error reading slice: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"\nSlice [{args.start}:{args.start + data.shape[0]}], shape={tuple(data.shape)}")
    if fmt == "fvecs":
        np.set_printoptions(precision=args.precision, suppress=True)
    # Print up to 3 vectors fully; then show only first few elements for the rest
    max_full = min(data.shape[0], 3)
    for i in range(max_full):
        print(f"#{args.start + i}: {data[i]}")
    if data.shape[0] > max_full:
        preview_len = min(10, data.shape[1])
        for i in range(max_full, data.shape[0]):
            row = data[i][:preview_len]
            ellipsis = " ..." if preview_len < data.shape[1] else ""
            print(f"#{args.start + i}: {row}{ellipsis}")

    if args.stats:
        if fmt == "fvecs":
            norms = np.linalg.norm(data, axis=1)
            print("\nStats (fvecs slice):")
            print(f"  min={data.min():.6f} max={data.max():.6f} mean={data.mean():.6f}")
            print(f"  L2-norms -> min={norms.min():.6f} max={norms.max():.6f} mean={norms.mean():.6f}")
        else:
            print("\nStats (ivecs slice):")
            print(f"  min={int(data.min())} max={int(data.max())} mean={float(data.mean()):.6f}")

if __name__ == "__main__":
    main()