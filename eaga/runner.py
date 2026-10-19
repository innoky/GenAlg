import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.setdefault("name", Path(path).stem)
    return cfg


def run_series(cfg: dict[str, Any], seeds: list[int],
               run_one: Callable[[dict, int], tuple[dict, list[dict]]],
               out_dir: str | Path, jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            results = list(ex.map(partial(run_one, cfg), seeds))
    else:
        results = [run_one(cfg, s) for s in seeds]
    rows, hist, extra = [], [], []
    for seed, res in zip(seeds, results):
        row, history = res[0], res[1]
        rows.append({"config": cfg["name"], "seed": seed, **row})
        hist.extend({"config": cfg["name"], "seed": seed, **h} for h in history)
        if len(res) > 2:
            extra.extend({"config": cfg["name"], "seed": seed, **e} for e in res[2])
    if extra:
        pd.DataFrame(extra).to_csv(out / "extra.csv", index=False, float_format="%.10g")
    runs = pd.DataFrame(rows)
    history_df = pd.DataFrame(hist)
    runs.to_csv(out / "runs.csv", index=False, float_format="%.10g")
    history_df.to_csv(out / "history.csv", index=False, float_format="%.6g")
    with open(out / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    return runs, history_df


def summarize(runs: pd.DataFrame, column: str, minimize: bool = True,
              by: str = "config") -> pd.DataFrame:
    g = runs.groupby(by, sort=False)[column]
    best, worst = (g.min(), g.max()) if minimize else (g.max(), g.min())
    return pd.DataFrame({"runs": g.size(), "best": best, "mean": g.mean(),
                         "median": g.median(), "std": g.std(ddof=1),
                         "worst": worst}).reset_index()


def step_on_grid(evals: np.ndarray, values: np.ndarray,
                 grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(evals, grid, side="right") - 1
    out = np.full(grid.shape, np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out
