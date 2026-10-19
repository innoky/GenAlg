import argparse

import numpy as np

from eaga.rng import make_rng

from .knapsack import DEFAULT_INSTANCE, Instance

PARAMS = {"n": 40, "low": 5, "high": 60, "coef_mass": 0.8, "coef_volume": 0.8,
          "noise_sd": 12.0, "cap_ratio": 0.35}


def generate(seed: int = 527, n: int = PARAMS["n"]) -> Instance:
    p = PARAMS
    rng = make_rng(seed)
    mass = rng.integers(p["low"], p["high"] + 1, size=n)
    volume = rng.integers(p["low"], p["high"] + 1, size=n)
    noise = rng.normal(0.0, p["noise_sd"], size=n)
    value = np.maximum(1, np.round(p["coef_mass"] * mass + p["coef_volume"] * volume
                                   + noise)).astype(np.int64)
    W = int(round(p["cap_ratio"] * mass.sum()))
    V = int(round(p["cap_ratio"] * volume.sum()))
    meta = {"seed": seed, **{k: v for k, v in p.items() if k != "n"}, "n": n,
            "procedure": "w,v ~ U{5..60}; p = max(1, round(0.8w + 0.8v + N(0,12^2))); "
                         "W = round(0.35*sum w), V = round(0.35*sum v)"}
    return Instance(mass, volume, value, W, V, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=527)
    ap.add_argument("--out", default=str(DEFAULT_INSTANCE))
    a = ap.parse_args()
    inst = generate(a.seed)
    inst.to_json(a.out)
    print(f"n={inst.n}, W={inst.cap_mass} (Σw={inst.mass.sum()}), "
          f"V={inst.cap_volume} (Σv={inst.volume.sum()}), Σp={inst.value.sum()} -> {a.out}")


if __name__ == "__main__":
    main()
