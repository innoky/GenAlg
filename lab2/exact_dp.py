import time

import numpy as np

from .knapsack import Instance


def solve(inst: Instance) -> tuple[int, np.ndarray, float]:
    t0 = time.perf_counter()
    W, V = inst.cap_mass, inst.cap_volume
    # dp[w, v] = лучшая ценность при массе <= w и объёме <= v
    dp = np.zeros((W + 1, V + 1), dtype=np.int64)
    take = np.zeros((inst.n, W + 1, V + 1), dtype=bool)
    for i, (wi, vi, pi) in enumerate(zip(inst.mass, inst.volume, inst.value)):
        if wi > W or vi > V:
            continue
        cand = dp[:W + 1 - wi, :V + 1 - vi] + pi
        better = cand > dp[wi:, vi:]
        take[i, wi:, vi:] = better
        dp[wi:, vi:] = np.where(better, cand, dp[wi:, vi:])
    x = np.zeros(inst.n, dtype=np.int8)
    w, v = W, V
    for i in range(inst.n - 1, -1, -1):
        if take[i, w, v]:
            x[i] = 1
            w -= inst.mass[i]
            v -= inst.volume[i]
    return int(dp[W, V]), x, time.perf_counter() - t0
