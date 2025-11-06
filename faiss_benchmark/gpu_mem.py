import os

def _nvidia_smi_available():
    return os.system("which nvidia-smi > /dev/null 2>&1") == 0

def _read_pynvml():
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "total_bytes": int(mem.total),
            "used_bytes": int(mem.used),
            "free_bytes": int(mem.free),
        }
    except Exception:
        return None

def _read_nvidia_smi():
    try:
        # Query memory for GPU 0 only; returns in MiB
        import subprocess
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits"
        ], text=True)
        line = out.strip().splitlines()[0]
        total_mib, used_mib, free_mib = [int(x.strip()) for x in line.split(',')]
        mib = 1024 * 1024
        return {
            "total_bytes": total_mib * mib,
            "used_bytes": used_mib * mib,
            "free_bytes": free_mib * mib,
        }
    except Exception:
        return None

def get_gpu_memory():
    """Return GPU 0 memory usage dict in bytes or None if unavailable."""
    info = _read_pynvml()
    if info:
        return info
    if _nvidia_smi_available():
        return _read_nvidia_smi()
    return None

def format_gpu_memory(info):
    if not info:
        return "GPU memory: N/A"
    def to_gb(b):
        return b / (1024 ** 3)
    return (
        f"GPU memory total={to_gb(info['total_bytes']):.2f}GB, "
        f"used={to_gb(info['used_bytes']):.2f}GB, "
        f"free={to_gb(info['free_bytes']):.2f}GB"
    )