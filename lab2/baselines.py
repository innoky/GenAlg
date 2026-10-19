import time

import numpy as np

from eaga.core import CountedFunction
from eaga.rng import make_rng

from .knapsack import Instance


def greedy(inst: Instance) -> np.ndarray:
    x = np.zeros(inst.n, dtype=np.int8)
    m = v = 0
    for i in np.argsort(-inst.efficiency(), kind="stable"):
        if m + inst.mass[i] <= inst.cap_mass and v + inst.volume[i] <= inst.cap_volume:
            x[i] = 1
            m += inst.mass[i]
            v += inst.volume[i]
    return x


def random_feasible(rng: np.random.Generator, inst: Instance, k: int) -> np.ndarray:
    X = np.zeros((k, inst.n), dtype=np.int8)
    for row in X:
        m = v = 0
        for i in rng.permutation(inst.n):
            if m + inst.mass[i] <= inst.cap_mass and v + inst.volume[i] <= inst.cap_volume:
                row[i] = 1
                m += inst.mass[i]
                v += inst.volume[i]
    return X


def run_random_feasible(cfg: dict, seed: int, inst: Instance | None = None,
                        opt: int | None = None) -> tuple[dict, list[dict]]:
    from .ga import KnapsackObjective
    inst = inst or Instance.from_json(cfg.get("instance", "lab2/data/instance.json"))
    rng = make_rng(seed)
    obj = KnapsackObjective(inst, "repair", opt=opt)
    f = CountedFunction(obj, cfg["budget"])
    obj.counter = f
    batch = cfg.get("batch", 48)
    t0 = time.perf_counter()
    history, b = [], 0
    while f.remaining > 0:
        f(random_feasible(rng, inst, min(batch, f.remaining)))
        history.append({"gen": b, "evals": f.evals, "best_so_far": f.best_cost,
                        "best_feasible_so_far": obj.best_feasible_value})
        b += 1
    row = {"best_value": obj.best_feasible_value, "evals": f.evals,
           "time_s": time.perf_counter() - t0, "generations": None,
           "final_best_feasible": True, "final_best_value": obj.best_feasible_value,
           "final_pop_feasible_frac": 1.0, "feasible_evaluated_frac": 1.0,
           "opt_hit_eval": obj.opt_hit_eval,
           "best_x": "".join(map(str, obj.best_feasible_x.tolist()))}
    return row, history
