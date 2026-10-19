import numpy as np


def tournament(rng: np.random.Generator, costs: np.ndarray, n: int,
               k: int = 3) -> np.ndarray:
    entrants = rng.integers(0, len(costs), size=(n, k))
    winners = np.argmin(costs[entrants], axis=1)
    return entrants[np.arange(n), winners]


def linear_ranking(rng: np.random.Generator, costs: np.ndarray, n: int,
                   s: float = 1.5) -> np.ndarray:
    N = len(costs)
    order = np.argsort(costs, kind="stable")
    rank = np.arange(N)[::-1]
    p = ((2 - s) + 2 * (s - 1) * rank / (N - 1)) / N
    cum = np.cumsum(p)
    cum[-1] = 1.0
    pointers = (rng.random() + np.arange(n)) / n
    picked = order[np.searchsorted(cum, pointers, side="right")]
    return rng.permutation(picked)


SELECTORS = {"tournament": tournament, "ranking": linear_ranking}
