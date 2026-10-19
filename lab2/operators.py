import numpy as np


def init_bernoulli(rng: np.random.Generator, k: int, n: int, p: float = 0.5) -> np.ndarray:
    return (rng.random((k, n)) < p).astype(np.int8)


def uniform_crossover(rng: np.random.Generator, A: np.ndarray, B: np.ndarray):
    mask = rng.random(A.shape) < 0.5
    return np.where(mask, A, B), np.where(mask, B, A)


def two_point_crossover(rng: np.random.Generator, A: np.ndarray, B: np.ndarray):
    k, n = A.shape
    cuts = np.sort(np.stack([rng.integers(0, n + 1, size=k),
                             rng.integers(0, n + 1, size=k)], axis=1), axis=1)
    idx = np.arange(n)
    mask = (idx >= cuts[:, :1]) & (idx < cuts[:, 1:])
    return np.where(mask, B, A), np.where(mask, A, B)


def bitflip(rng: np.random.Generator, X: np.ndarray, pm: float) -> np.ndarray:
    flip = rng.random(X.shape) < pm
    return np.where(flip, 1 - X, X).astype(np.int8)


CROSSOVERS = {"uniform": uniform_crossover, "two_point": two_point_crossover}
