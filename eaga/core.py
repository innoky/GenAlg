import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class BudgetExhausted(RuntimeError):
    pass
class CountedFunction:

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], budget: int,
                 target: float | None = None):
        self.func = func
        self.budget = int(budget)
        self.target = target
        self.evals = 0
        self.best_cost = np.inf
        self.best_x: np.ndarray | None = None
        self.first_hit_eval: int | None = None

    @property
    def remaining(self) -> int:
        return self.budget - self.evals

    def __call__(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if n > self.remaining:
            raise BudgetExhausted(f"нужно {n} вызовов, осталось {self.remaining}")
        costs = np.asarray(self.func(X), dtype=float)
        assert costs.shape == (n,)
        if self.target is not None and self.first_hit_eval is None:
            hit = np.flatnonzero(costs < self.target)
            if hit.size:
                self.first_hit_eval = self.evals + int(hit[0]) + 1
        i = int(np.argmin(costs))
        if costs[i] < self.best_cost:
            self.best_cost = float(costs[i])
            self.best_x = X[i].copy()
        self.evals += n
        return costs


@dataclass
class GAResult:
    best_cost: float
    best_x: np.ndarray
    evals: int
    generations: int
    first_hit_eval: int | None
    first_hit_gen: int | None
    time_s: float
    history: list[dict[str, Any]] = field(default_factory=list)
    final_pop: np.ndarray | None = None
    final_costs: np.ndarray | None = None


def run_ga(f: CountedFunction, rng: np.random.Generator, *, pop_size: int,
           n_elite: int, init: Callable, select: Callable, vary: Callable,
           log_extra: Callable | None = None) -> GAResult:
    t0 = time.perf_counter()
    pop = init(rng, pop_size)
    costs = f(pop)
    gen = 0
    first_hit_gen = 0 if f.first_hit_eval is not None else None
    history = [_log(gen, f, pop, costs, log_extra)]
    n_off = pop_size - n_elite
    while f.remaining > 0:
        gen += 1
        order = np.argsort(costs, kind="stable")
        parent_idx = select(rng, costs, n_off)
        children = vary(rng, pop[parent_idx])
        k = min(n_off, f.remaining)
        children = children[:k]
        child_costs = f(children)
        keep = order[:pop_size - k]  # элита
        pop = np.concatenate([pop[keep], children])
        costs = np.concatenate([costs[keep], child_costs])
        if first_hit_gen is None and f.first_hit_eval is not None:
            first_hit_gen = gen
        history.append(_log(gen, f, pop, costs, log_extra))
    return GAResult(best_cost=f.best_cost, best_x=f.best_x, evals=f.evals,
                    generations=gen, first_hit_eval=f.first_hit_eval,
                    first_hit_gen=first_hit_gen,
                    time_s=time.perf_counter() - t0, history=history,
                    final_pop=pop, final_costs=costs)


def _log(gen, f, pop, costs, log_extra):
    row = {"gen": gen, "evals": f.evals, "best_so_far": f.best_cost,
           "pop_best": float(np.min(costs)), "pop_mean": float(np.mean(costs)),
           "pop_median": float(np.median(costs))}
    if log_extra is not None:
        row.update(log_extra(pop, costs))
    return row
