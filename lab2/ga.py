from functools import partial

import numpy as np

from eaga.core import CountedFunction, run_ga
from eaga.rng import make_rng
from eaga.selection import SELECTORS

from . import operators as ops
from .constraints import GreedyRepair, penalized_fitness
from .knapsack import Instance


class KnapsackObjective:
    def __init__(self, inst: Instance, mode: str, lam: float = 1.0,
                 opt: int | None = None):
        self.inst, self.mode, self.lam, self.opt = inst, mode, lam, opt
        self.counter: CountedFunction | None = None
        self.best_feasible_value = -1
        self.best_feasible_x: np.ndarray | None = None
        self.opt_hit_eval: int | None = None
        self.n_feasible_evaluated = 0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        feas = self.inst.feasible(X)
        _, _, val = self.inst.loads(X)
        if self.mode == "repair":
            assert feas.all(), "ремонт обязан давать допустимые решения"
            F = val.astype(float)
        else:
            F = penalized_fitness(self.inst, X, self.lam)
        self.n_feasible_evaluated += int(feas.sum())
        if feas.any():
            fv = np.where(feas, val, -1)
            i = int(np.argmax(fv))
            if fv[i] > self.best_feasible_value:
                self.best_feasible_value = int(fv[i])
                self.best_feasible_x = np.array(X[i], copy=True)
            if self.opt is not None and self.opt_hit_eval is None:
                hit = np.flatnonzero(fv >= self.opt)
                if hit.size:
                    self.opt_hit_eval = self.counter.evals + int(hit[0]) + 1
        return -F


def run_one(cfg: dict, seed: int, inst: Instance | None = None,
            opt: int | None = None) -> tuple[dict, list[dict]]:
    inst = inst or Instance.from_json(cfg.get("instance", "lab2/data/instance.json"))
    rng = make_rng(seed)
    n = inst.n
    ch = cfg["constraints"]
    repair = GreedyRepair(inst) if ch["type"] == "repair" else None
    obj = KnapsackObjective(inst, ch["type"], ch.get("lambda", 1.0), opt)
    f = CountedFunction(obj, cfg["budget"])
    obj.counter = f
    sel = cfg["selection"]
    select = partial(SELECTORS[sel["type"]], **{k: v for k, v in sel.items() if k != "type"})
    crossover = ops.CROSSOVERS[cfg["crossover"]["type"]]
    pc = cfg["crossover"]["pc"]
    pm = 1.0 / n if cfg["mutation"]["pm"] == "1/n" else cfg["mutation"]["pm"]

    def init(rng, k):
        X = ops.init_bernoulli(rng, k, n, cfg.get("init_p", 0.5))
        return repair(X) if repair else X

    def vary(rng, P):
        k = len(P)
        if k % 2:
            P = np.vstack([P, P[:1]])
        A, B = P[0::2], P[1::2]
        do = rng.random(len(A)) < pc
        C1, C2 = A.copy(), B.copy()
        if do.any():
            C1[do], C2[do] = crossover(rng, A[do], B[do])
        C = np.empty_like(P)
        C[0::2], C[1::2] = C1, C2
        C = ops.bitflip(rng, C, pm)
        if repair:
            C = repair(C)
        return C[:k]

    def log_extra(pop, costs):
        feas = inst.feasible(pop)
        return {"best_feasible_so_far": obj.best_feasible_value,
                "pop_feasible_frac": float(feas.mean())}

    res = run_ga(f, rng, pop_size=cfg["pop_size"], n_elite=cfg["n_elite"],
                 init=init, select=select, vary=vary, log_extra=log_extra)
    final_best = res.final_pop[np.argmin(res.final_costs)]
    fb_feas = bool(inst.feasible(final_best)[0])
    row = {"best_value": obj.best_feasible_value if obj.best_feasible_value >= 0 else None,
           "evals": res.evals,
           "time_s": res.time_s, "generations": res.generations,
           "final_best_feasible": fb_feas,
           "final_best_value": int(inst.loads(final_best)[2][0]),
           "final_best_x": "".join(map(str, final_best.tolist())),
           "final_pop_feasible_frac": float(inst.feasible(res.final_pop).mean()),
           "feasible_evaluated_frac": obj.n_feasible_evaluated / res.evals,
           "opt_hit_eval": obj.opt_hit_eval,
           "best_x": "".join(map(str, obj.best_feasible_x.tolist()))
           if obj.best_feasible_x is not None else ""}
    return row, res.history
