import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .runner import step_on_grid

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
          "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]


def convergence_bands(history: pd.DataFrame, grid: np.ndarray,
                      column: str = "best_so_far") -> dict[str, np.ndarray]:
    curves = []
    for _, h in history.groupby("seed"):
        h = h.sort_values("evals")
        curves.append(step_on_grid(h["evals"].to_numpy(), h[column].to_numpy(), grid))
    C = np.vstack(curves)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {"min": np.nanmin(C, 0), "median": np.nanmedian(C, 0),
                "mean": np.nanmean(C, 0), "max": np.nanmax(C, 0)}


def plot_convergence(histories: dict[str, pd.DataFrame], budget: int,
                     path: str | Path, *, title: str, ylabel: str,
                     logy: bool = True, column: str = "best_so_far",
                     floor: float = 1e-300, hline: float | None = None,
                     hline_label: str | None = None) -> None:
    start = min(int(h["evals"].min()) for h in histories.values())
    grid = np.unique(np.concatenate([np.geomspace(start, budget, 400).round(),
                                     [budget]])).astype(int)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (label, h), c in zip(histories.items(), COLORS):
        b = convergence_bands(h, grid, column)
        lo, hi = b["min"], b["max"]
        if logy:
            lo, hi = np.maximum(lo, floor), np.maximum(hi, floor)
        ax.fill_between(grid, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(grid, np.maximum(b["median"], floor) if logy else b["median"],
                color=c, lw=1.8, label=f"{label}: медиана (полоса мин–макс)")
        ax.plot(grid, np.maximum(b["mean"], floor) if logy else b["mean"],
                color=c, lw=1, ls="--")
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.8, ls=":", label=hline_label)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("число вызовов фитнес-функции")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(штрих — среднее по запускам)", fontsize=11)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_box(runs: pd.DataFrame, column: str, order: list[str], path: str | Path,
             *, title: str, ylabel: str, logy: bool = True,
             labels: list[str] | None = None, hline: float | None = None,
             hline_label: str | None = None, floor: float = 1e-300) -> None:
    data = []
    for name in order:
        v = runs.loc[runs["config"] == name, column].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        data.append(np.maximum(v, floor) if logy else v)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(order)), 5))
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(order) + 1), labels or order, rotation=15, fontsize=8)
    for i, v in enumerate(data, start=1):
        ax.plot(np.full(v.shape, i) + np.linspace(-0.12, 0.12, len(v)), v, ".",
                color="k", alpha=0.5, ms=4)
    if hline is not None:
        ax.axhline(hline, color="r", lw=0.8, ls=":", label=hline_label)
        ax.legend(fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
