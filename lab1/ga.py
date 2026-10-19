from functools import partial

import numpy as np

from eaga.core import CountedFunction, run_ga
from eaga.rng import make_rng
from eaga.selection import SELECTORS

from . import operators as ops
from .functions import DIM, LOWER, UPPER, sphere

BOUNDS = {"reflect": ops.reflect, "clip": ops.clip}


def build(cfg: dict):
    d = cfg.get("dim", DIM)
    lo, hi = cfg.get("lower", LOWER), cfg.get("upper", UPPER)
    mut = cfg["mutation"]
    adaptive = mut["type"] == "gauss_selfadaptive"
    sel = cfg["selection"]
    select = partial(SELECTORS[sel["type"]],
                     **{k: v for k, v in sel.items() if k != "type"})
    cx = cfg["crossover"]
    bound = BOUNDS[cfg["bounds"]]
    pm = mut["pm"] if mut["pm"] != "1/d" else 1.0 / d

    def init(rng, n):
        X = ops.init_uniform(rng, n, d, lo, hi)
        if adaptive:
            X = np.hstack([X, np.full((n, 1), mut["sigma0"])])
        return X

    def vary(rng, P):
        n = len(P)
        m = n + (n % 2)
        if m != n:
            P = np.vstack([P, P[:1]])
        A, B = P[0::2], P[1::2]
        do_cx = rng.random(len(A)) < cx["pc"]
        C1, C2 = A.copy(), B.copy()
        if do_cx.any():
            x1, x2 = ops.blx_alpha(rng, A[do_cx, :d], B[do_cx, :d], cx["alpha"])
            C1[do_cx, :d], C2[do_cx, :d] = x1, x2
            if adaptive:
                s = np.sqrt(A[do_cx, d] * B[do_cx, d])
                C1[do_cx, d], C2[do_cx, d] = s, s
        C = np.empty_like(P)
        C[0::2], C[1::2] = C1, C2
        if adaptive:
            tau = mut.get("tau", 1.0 / np.sqrt(d))
            C[:, d] = ops.lognormal_sigma(rng, C[:, d], tau, mut["sigma_min"],
                                          mut.get("sigma_max", hi - lo))
            C[:, :d] = ops.gauss_mutation(rng, C[:, :d], C[:, d], pm)
        else:
            C = ops.gauss_mutation(rng, C, mut["sigma"], pm)
        C[:, :d] = bound(C[:, :d], lo, hi)
        return C[:n]

    def log_extra(pop, costs):
        row = {"diversity": float(np.mean(np.std(pop[:, :d], axis=0)))}
        if adaptive:
            row["sigma_median"] = float(np.median(pop[:, d]))
            row["sigma_best"] = float(pop[np.argmin(costs), d])
        return row

    return init, select, vary, log_extra, adaptive, d


def run_one(cfg: dict, seed: int) -> tuple[dict, list[dict]]:
    rng = make_rng(seed)
    init, select, vary, log_extra, adaptive, d = build(cfg)
    f = CountedFunction(lambda X: sphere(X[:, :d]), cfg["budget"],
                        target=cfg.get("target"))
    res = run_ga(f, rng, pop_size=cfg["pop_size"], n_elite=cfg["n_elite"],
                 init=init, select=select, vary=vary, log_extra=log_extra)
    row = {"best_f": res.best_cost, "evals": res.evals, "time_s": res.time_s,
           "generations": res.generations,
           "hit_eval": res.first_hit_eval, "hit_gen": res.first_hit_gen,
           "success": bool(cfg.get("target") is not None
                           and res.best_cost < cfg["target"]),
           "best_x": " ".join(f"{v:.3e}" for v in res.best_x[:d])}
    if adaptive:
        row["final_sigma_median"] = float(np.median(res.final_pop[:, d]))
    return row, res.history
