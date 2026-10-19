import json
from functools import partial

import numpy as np

from eaga.core import CountedFunction, run_ga
from eaga.rng import make_rng
from eaga.selection import SELECTORS

from . import operators as ops
from .data_gen import load
from .tree import Generator, depth, evaluate, to_str

BAD = 1e10  # ошибка для деревьев, дающих nan/inf


def nrmse(tree, X, y):
    pred = evaluate(tree, X)
    if not np.all(np.isfinite(pred)):
        return BAD
    return float(np.sqrt(np.mean((pred - y) ** 2)) / np.std(y))


class Objective:
    """cost = NRMSE на train + alpha * размер; попутно ведёт архив
    недоминируемых по (NRMSE, размер) деревьев."""

    def __init__(self, X, y, alpha):
        self.X, self.y, self.alpha = X, y, alpha
        self.archive = {}  # размер -> (nrmse, дерево)

    def __call__(self, pop):
        costs = np.empty(len(pop))
        for k, tree in enumerate(pop):
            err = nrmse(tree, self.X, self.y)
            costs[k] = err + self.alpha * len(tree)
            old = self.archive.get(len(tree))
            if old is None or err < old[0]:
                self.archive[len(tree)] = (err, list(tree))
        return costs

    def front(self):
        out, best = [], np.inf
        for size in sorted(self.archive):
            err, tree = self.archive[size]
            if err < best:
                out.append((size, err, tree))
                best = err
        return out


def load_data(cfg):
    folder = cfg.get("data", "lab3/data")
    return {k: load(k, folder) for k in ("train", "test", "extrapolation")}


def run_one(cfg, seed):
    data = load_data(cfg)
    X, y = data["train"]
    rng = make_rng(seed)
    gen = Generator(X.shape[1])
    obj = Objective(X, y, cfg["alpha"])
    f = CountedFunction(obj, cfg["budget"])
    max_depth = cfg["max_depth"]
    sel = cfg["selection"]
    select = partial(SELECTORS[sel["type"]], **{k: v for k, v in sel.items() if k != "type"})

    def init(rng, n):
        return gen.ramped(rng, n, *cfg["init_depth"])

    def vary(rng, P):
        n = len(P)
        if cfg["algorithm"] == "random":
            return init(rng, n)
        C = np.empty(n + n % 2, dtype=object)
        for k in range(0, n, 2):
            a, b = P[k], P[(k + 1) % n]
            if rng.random() < cfg["pc"]:
                a, b = ops.subtree_crossover(rng, a, b, max_depth)
            C[k], C[k + 1] = list(a), list(b)
        for k in range(n):
            if rng.random() < cfg["p_subtree"]:
                C[k] = ops.subtree_mutation(rng, C[k], gen, max_depth)
            if rng.random() < cfg["p_point"]:
                C[k] = ops.point_mutation(rng, C[k], gen)
        return C[:n]

    def log_extra(pop, costs):
        sizes = np.array([len(t) for t in pop])
        b = pop[np.argmin(costs)]
        return {"best_size": len(b), "best_train": nrmse(b, X, y),
                "size_mean": float(sizes.mean()), "size_max": int(sizes.max())}

    res = run_ga(f, rng, pop_size=cfg["pop_size"], n_elite=cfg["n_elite"],
                 init=init, select=select, vary=vary, log_extra=log_extra)
    best = res.best_x
    row = {"cost": res.best_cost, "train": nrmse(best, *data["train"]),
           "test": nrmse(best, *data["test"]),
           "extrapolation": nrmse(best, *data["extrapolation"]),
           "size": len(best), "depth": depth(best), "evals": res.evals,
           "generations": res.generations, "time_s": res.time_s,
           "solved": nrmse(best, *data["test"]) < cfg["solved_test_nrmse"],
           "expr": to_str(best), "tree": json.dumps(best)}
    front = [{"size": s, "train": e, "test": nrmse(t, *data["test"]),
              "extrapolation": nrmse(t, *data["extrapolation"]),
              "expr": to_str(t), "tree": json.dumps(t)} for s, e, t in obj.front()]
    return row, res.history, front
