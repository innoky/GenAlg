import numpy as np


def variant(N: int = 20, Q: int = 2, Y: int = 26) -> tuple[int, int]:
    V = 1 + ((17 * N + 7 * Q + Y) % 20)
    return V, 5 + (V % 6)


V, DIM = variant()
LOWER, UPPER = -100.0, 100.0
F_OPT = 0.0


def sphere(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    return np.einsum("ij,ij->i", X, X)
