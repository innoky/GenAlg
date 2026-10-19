import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eaga.rng import make_rng

DATA = Path(__file__).parent / "data"
TARGET = "x0^2 + x0*sin(x1) + 0.5*x1"


def target(X):
    return X[:, 0] ** 2 + X[:, 0] * np.sin(X[:, 1]) + 0.5 * X[:, 1]


def generate(seed=327, n_train=200, n_test=200, n_extra=200):
    rng = make_rng(seed)
    sets = {"train": rng.uniform(-3, 3, (n_train, 2)),
            "test": rng.uniform(-3, 3, (n_test, 2)),
            "extrapolation": rng.uniform(-5, 5, (n_extra, 2))}
    return {k: pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "y": target(X)})
            for k, X in sets.items()}


def load(name, folder=DATA):
    df = pd.read_csv(Path(folder) / f"{name}.csv")
    return df[["x0", "x1"]].to_numpy(), df["y"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=327)
    ap.add_argument("--out", default=str(DATA))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name, df in generate(a.seed).items():
        df.to_csv(Path(a.out) / f"{name}.csv", index=False, float_format="%.17g")
        print(f"{name}: {len(df)} точек, std(y) = {df.y.std(ddof=0):.4f}")


if __name__ == "__main__":
    main()
