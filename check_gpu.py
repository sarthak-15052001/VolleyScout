#!/usr/bin/env python3
"""Quick GPU check for conda py312 — run in your normal terminal (not Cursor agent)."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch not installed in this Python:", sys.executable)
        return 1

    print("python     :", sys.executable)
    print("torch      :", torch.__version__)
    print("cuda built :", torch.version.cuda)
    print("cuda avail :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("FAIL: GPU not visible to PyTorch. Check nvidia-smi in this same shell.")
        return 1

    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"gpu[0]    : {name} ({mem:.1f} GiB)")
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul     : OK")
    print("PASS: VolleyScout can use this GPU (device=0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
