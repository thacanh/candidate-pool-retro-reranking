"""Recommend a conservative AiZynthFinder process count for one host."""

from __future__ import annotations

import os

import psutil


def available_cpu_count() -> int:
    try:
        affinity = psutil.Process().cpu_affinity()
        if affinity:
            return len(affinity)
    except (AttributeError, NotImplementedError, psutil.Error):
        pass
    return os.cpu_count() or 1


def recommend_workers() -> int:
    logical_cpus = available_cpu_count()
    ram_gib = psutil.virtual_memory().total / (1024**3)
    # Reserve about 6 GiB for the OS, stock/config, parent and file cache.
    # Each independent policy process is budgeted at 2.25 GiB.  CPU is capped
    # at half the available logical threads because RDKit/ONNX may use helper
    # threads despite the outer thread limits.
    memory_limit = max(1, int((ram_gib - 6.0) // 2.25))
    cpu_limit = max(1, logical_cpus // 2)
    return max(1, min(8, memory_limit, cpu_limit))


if __name__ == "__main__":
    print(recommend_workers())
