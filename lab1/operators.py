import numpy as np


def init_uniform(rng: np.random.Generator, n: int, d: int, lo: float,
                 hi: float) -> np.ndarray:
    return rng.uniform(lo, hi, size=(n, d))


def blx_alpha(rng: np.random.Generator, P1: np.ndarray, P2: np.ndarray,
              alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.minimum(P1, P2), np.maximum(P1, P2)
    ext = alpha * (hi - lo)
    a, b = lo - ext, hi + ext
    C1 = a + rng.random(P1.shape) * (b - a)
    C2 = a + rng.random(P1.shape) * (b - a)
    return C1, C2


def reflect(X: np.ndarray, lo: float, hi: float) -> np.ndarray:
    # отражаем только вышедшие гены: формула через mod округляет x до ~1e-14
    out = (X < lo) | (X > hi)
    if not out.any():
        return X
    r = hi - lo
    Y = X.copy()
    t = np.mod(X[out] - lo, 2 * r)
    Y[out] = lo + np.where(t > r, 2 * r - t, t)
    return Y


def clip(X: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(X, lo, hi)


def gauss_mutation(rng: np.random.Generator, X: np.ndarray, sigma,
                   pm: float) -> np.ndarray:
    mask = rng.random(X.shape) < pm
    sig = np.asarray(sigma, dtype=float)
    if sig.ndim == 1:
        sig = sig[:, None]
    return X + mask * sig * rng.standard_normal(X.shape)


def lognormal_sigma(rng: np.random.Generator, sigma: np.ndarray, tau: float,
                    smin: float, smax: float) -> np.ndarray:
    return np.clip(sigma * np.exp(tau * rng.standard_normal(sigma.shape)), smin, smax)
