import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eaga.plotting import COLORS, convergence_bands, plot_box, plot_convergence
from eaga.runner import load_config, parse_seeds, run_series, summarize

from . import ga, random_search
from .functions import DIM, LOWER, UPPER

TARGET = 1e-8
BUDGET = 100_000
PLOT_FLOOR = 1e-30
LABELS = {
    "base_fixed": "ГА, σ=2 фикс., pm=1/d, pop50 (база)",
    "selfadaptive_sigma": "ГА, σ самоадапт., pm=1/d",
    "fixed_pm1": "ГА, σ=2 фикс., pm=1",
    "selfadaptive_pm1": "ГА, σ самоадапт., pm=1",
    "pop30": "ГА, pop30",
    "pop100": "ГА, pop100",
    "ranking": "ГА, ранговая селекция",
    "random_search": "случайный поиск",
}
ORDER = list(LABELS)


def run_configs(paths: list[str], seeds: list[int], out: Path) -> None:
    for p in paths:
        cfg = load_config(p)
        module = random_search if cfg["algorithm"] == "random" else ga
        runs, _ = run_series(cfg, seeds, module.run_one, out)
        print(f"{cfg['name']:>20}: median best_f = {runs['best_f'].median():.3e}, "
              f"успехов {int(runs['success'].sum())}/{len(runs)}")


def load_all(out: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    runs, hist = [], {}
    for name in ORDER:
        d = out / name
        if (d / "runs.csv").exists():
            runs.append(pd.read_csv(d / "runs.csv"))
            hist[name] = pd.read_csv(d / "history.csv")
    return pd.concat(runs, ignore_index=True), hist


def write_summary(runs: pd.DataFrame, out: Path) -> pd.DataFrame:
    s = summarize(runs, "best_f", minimize=True)
    g = runs.groupby("config", sort=False)
    s["success_rate"] = g["success"].mean().to_numpy()
    s["evals_mean"] = g["evals"].mean().to_numpy()
    s["time_s_mean"] = g["time_s"].mean().to_numpy()
    s["hit_eval_median"] = g["hit_eval"].median().to_numpy()
    s.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    with open(out / "summary.md", "w", encoding="utf-8") as fh:
        fh.write("| конфигурация | запусков | best | mean | median | std | worst | "
                 "доля f<1e-8 | вызовов | время, с | медиана вызова достижения 1e-8 (по успешным) |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in s.iterrows():
            hit = "—" if pd.isna(r.hit_eval_median) else f"{r.hit_eval_median:.0f}"
            fh.write(f"| {r.config} | {r.runs} | {r.best:.3e} | {r['mean']:.3e} | "
                     f"{r['median']:.3e} | {r['std']:.3e} | {r.worst:.3e} | "
                     f"{r.success_rate:.2f} | {r.evals_mean:.0f} | {r.time_s_mean:.3f} | {hit} |\n")
    return s


def plots(runs: pd.DataFrame, hist: dict[str, pd.DataFrame], out: Path) -> None:
    lab = {k: LABELS[k] for k in hist}
    kw = dict(budget=BUDGET, floor=PLOT_FLOOR, hline=TARGET,
              ylabel=f"лучшее f(x), лог. шкала (значения < {PLOT_FLOOR:g} — на уровне {PLOT_FLOOR:g})",
              hline_label="порог 1e-8")
    plot_convergence({lab[k]: h for k, h in hist.items()}, path=out / "convergence_all.png",
                     title="Сфера d=6: сходимость всех конфигураций, 20 запусков", **kw)
    pick = lambda names: {lab[k]: hist[k] for k in names if k in hist}
    plot_convergence(pick(["base_fixed", "selfadaptive_sigma", "fixed_pm1",
                           "selfadaptive_pm1", "random_search"]),
                     path=out / "convergence_sigma.png",
                     title="Фиксированная σ против самоадаптивной", **kw)
    plot_convergence(pick(["pop30", "base_fixed", "pop100", "random_search"]),
                     path=out / "convergence_pop.png",
                     title="Размер популяции 30 / 50 / 100 (σ фикс.)", **kw)
    names = [k for k in ORDER if k in hist]
    plot_box(runs, "best_f", names, out / "boxplot_final.png",
             labels=[LABELS[k].replace(", ", ",\n") for k in names],
             title="Финальное лучшее f после 1e5 вызовов (20 запусков)",
             ylabel="f (лог. шкала)", hline=TARGET, hline_label="порог 1e-8",
             floor=PLOT_FLOOR)
    plot_sigma(hist, out / "sigma_adaptation.png")
    plot_random_theory(hist.get("random_search"), out / "random_search_theory.png")
    write_theory(runs, out / "theory.md")


def write_theory(runs: pd.DataFrame, path: Path) -> None:
    from math import gamma, pi
    c = pi ** (DIM / 2) / gamma(DIM / 2 + 1) / (UPPER - LOWER) ** DIM
    pred = random_search_expected(np.array([float(BUDGET)]))[0]
    obs = runs.loc[runs.config == "random_search", "best_f"].median()
    B_target = np.log(2) / (c * TARGET ** (DIM / 2))
    path.write_text(
        f"- Доля объёма области, где f < t: c·t^(d/2), c = π^3/3!/200^6 = {c:.4e}.\n"
        f"- Случайный поиск, B = {BUDGET}: теоретическая медиана min f = {pred:.1f}, "
        f"наблюдаемая медиана по 20 запускам = {obs:.1f}.\n"
        f"- Бюджет, при котором медиана случайного поиска достигла бы f < {TARGET:g}: "
        f"B ≈ ln2 / (c·t^3) = {B_target:.2e} вызовов.\n", encoding="utf-8")


def plot_sigma(hist: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    grid = np.unique(np.geomspace(50, BUDGET, 300).round()).astype(int)
    for name, c in zip(["selfadaptive_sigma", "selfadaptive_pm1"], COLORS[1:]):
        if name not in hist:
            continue
        h = hist[name]
        b = convergence_bands(h, grid, "sigma_median")
        axes[0].fill_between(grid, b["min"], b["max"], color=c, alpha=0.15)
        axes[0].plot(grid, b["median"], color=c, label=LABELS[name])
        f = convergence_bands(h, grid, "best_so_far")
        axes[1].plot(grid, np.maximum(f["median"], PLOT_FLOOR), color=c, label=LABELS[name])
        axes[1].plot(grid, np.sqrt(np.maximum(f["median"], PLOT_FLOOR) / DIM), color=c,
                     ls=":", label=f"√(f/d) — масштаб расстояния до x* ({name})")
    axes[0].axhline(2.0, color="k", ls="--", lw=0.8, label="фиксированная σ=2")
    axes[0].set_title("медиана σ в популяции (полоса мин–макс по запускам)", fontsize=10)
    axes[1].set_title("медиана лучшего f и √(f/d)", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("число вызовов")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def random_search_expected(B: np.ndarray, d: int = DIM, lo: float = LOWER,
                           hi: float = UPPER) -> np.ndarray:
    from math import gamma, pi
    c = pi ** (d / 2) / gamma(d / 2 + 1) / (hi - lo) ** d
    return (np.log(2) / (B * c)) ** (2 / d)


def plot_random_theory(h: pd.DataFrame | None, path: Path) -> None:
    if h is None:
        return
    grid = np.unique(np.geomspace(50, BUDGET, 200).round()).astype(int)
    b = convergence_bands(h, grid)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(grid, b["min"], b["max"], alpha=0.2, label="мин–макс, 20 запусков")
    ax.plot(grid, b["median"], label="медиана, эксперимент")
    ax.plot(grid, random_search_expected(grid.astype(float)), "k--",
            label="теория: медиана ∝ B^(−2/d) = B^(−1/3)")
    ax.axhline(TARGET, color="r", ls=":", lw=0.8, label="порог 1e-8")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("число вызовов B")
    ax.set_ylabel("лучшее f")
    ax.set_title("Случайный поиск на сфере d=6: f_best ∝ B^(−1/3)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=[], help="JSON-конфигурации")
    ap.add_argument("--seeds", default="0-19", help="напр. 0-19 или 0,1,5")
    ap.add_argument("--out", default="lab1/results")
    ap.add_argument("--plots-only", action="store_true",
                    help="только пересобрать сводку и графики из CSV")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        run_configs(args.config, parse_seeds(args.seeds), out)
    runs, hist = load_all(out)
    write_summary(runs, out)
    plots(runs, hist, out)
    print(f"сводка: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
