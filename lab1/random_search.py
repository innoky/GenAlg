import time

from eaga.core import CountedFunction
from eaga.rng import make_rng

from .functions import DIM, LOWER, UPPER, sphere


def run_one(cfg: dict, seed: int) -> tuple[dict, list[dict]]:
    rng = make_rng(seed)
    d, lo, hi = cfg.get("dim", DIM), cfg.get("lower", LOWER), cfg.get("upper", UPPER)
    f = CountedFunction(sphere, cfg["budget"], target=cfg.get("target"))
    batch = cfg.get("batch", 50)
    t0 = time.perf_counter()
    history, b = [], 0
    while f.remaining > 0:
        n = min(batch, f.remaining)
        f(rng.uniform(lo, hi, size=(n, d)))
        history.append({"gen": b, "evals": f.evals, "best_so_far": f.best_cost})
        b += 1
    row = {"best_f": f.best_cost, "evals": f.evals,
           "time_s": time.perf_counter() - t0, "generations": None,
           "hit_eval": f.first_hit_eval, "hit_gen": None,
           "success": bool(cfg.get("target") is not None and f.best_cost < cfg["target"]),
           "best_x": " ".join(f"{v:.3e}" for v in f.best_x)}
    return row, history
