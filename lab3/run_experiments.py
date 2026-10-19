import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eaga.plotting import COLORS, convergence_bands, plot_box
from eaga.runner import load_config, parse_seeds, run_series, summarize

from . import baselines, gp

BUDGET = 100_000
LABELS = {"alpha0": "ГП, α=0", "alpha1e-3": "ГП, α=1e-3", "alpha1e-2": "ГП, α=1e-2",
          "random_search": "случайные деревья"}
ORDER = list(LABELS)
EXACT = 1e-6


def run_configs(paths, seeds, out, jobs):
    for p in paths:
        cfg = load_config(p)
        runs, _ = run_series(cfg, seeds, gp.run_one, out, jobs=jobs)
        print(f"{cfg['name']:>14}: median test = {runs['test'].median():.3e}, "
              f"решено {int(runs['solved'].sum())}/{len(runs)}")


def load_all(out):
    runs, hist, fronts = [], {}, {}
    for name in ORDER:
        d = out / name
        if (d / "runs.csv").exists():
            runs.append(pd.read_csv(d / "runs.csv"))
            hist[name] = pd.read_csv(d / "history.csv")
            fronts[name] = pd.read_csv(d / "extra.csv")
    return pd.concat(runs, ignore_index=True), hist, fronts


def write_baselines(out):
    rows = baselines.evaluate(gp.load_data({}))
    pd.DataFrame(rows).to_csv(out / "baselines.csv", index=False, float_format="%.6g")
    lines = ["| модель | параметров | train NRMSE | test NRMSE | экстраполяция NRMSE |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['params']} | {r['train']:.4f} | {r['test']:.4f} | "
              f"{r['extrapolation']:.4f} |" for r in rows]
    (out / "baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(runs, out):
    runs = runs.assign(exact=runs["test"] < EXACT)
    s = summarize(runs, "test", minimize=True)
    g = runs.groupby("config", sort=False)
    s["solved"] = g["solved"].sum().to_numpy()
    s["exact"] = g["exact"].sum().to_numpy()
    s["train_median"] = g["train"].median().to_numpy()
    s["extrapolation_median"] = g["extrapolation"].median().to_numpy()
    s["size_median"] = g["size"].median().to_numpy()
    s["size_max"] = g["size"].max().to_numpy()
    s["time_s_mean"] = g["time_s"].mean().to_numpy()
    s["evals_mean"] = g["evals"].mean().to_numpy()
    s.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    lines = ["| конфигурация | запусков | test best | test mean | test median | test std | "
             "test worst | решено (test < 0.01) | точно (test < 1e-6) | train median | "
             "экстраполяция median | размер median | размер max | вызовов | время, с |",
             "|" + "---|" * 15]
    for _, r in s.iterrows():
        lines.append(f"| {r.config} | {r.runs} | {r.best:.2e} | {r['mean']:.2e} | "
                     f"{r['median']:.2e} | {r['std']:.2e} | {r.worst:.2e} | "
                     f"{int(r.solved)}/{r.runs} | {int(r.exact)}/{r.runs} | "
                     f"{r.train_median:.2e} | {r.extrapolation_median:.2e} | "
                     f"{r.size_median:.0f} | {r.size_max:.0f} | {r.evals_mean:.0f} | "
                     f"{r.time_s_mean:.1f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def permutation_test(a, b, n_perm=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        hits += abs(p[:len(a)].mean() - p[len(a):].mean()) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def write_comparisons(runs, out):
    pairs = [("alpha0", "alpha1e-3"), ("alpha1e-3", "alpha1e-2"), ("alpha0", "alpha1e-2"),
             ("alpha1e-3", "random_search")]
    lines = ["| A | B | показатель | median A | median B | p-value |", "|---|---|---|---|---|---|"]
    for a, b in pairs:
        for col, transform in [("test", np.log10), ("size", lambda v: v)]:
            va = runs.loc[runs.config == a, col].to_numpy(float)
            vb = runs.loc[runs.config == b, col].to_numpy(float)
            if len(va) and len(vb):
                p = permutation_test(transform(np.maximum(va, 1e-300)),
                                     transform(np.maximum(vb, 1e-300)))
                name = "log10 test NRMSE" if col == "test" else "размер дерева"
                lines.append(f"| {a} | {b} | {name} | {np.median(va):.3g} | "
                             f"{np.median(vb):.3g} | {p:.4f} |")
    (out / "comparisons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_convergence(hist, out):
    grid = np.unique(np.geomspace(500, BUDGET, 300).round()).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (name, h), c in zip(hist.items(), COLORS):
        b = convergence_bands(h, grid, "best_train")
        axes[0].fill_between(grid, np.maximum(b["min"], 1e-17), b["max"], color=c, alpha=0.12, lw=0)
        axes[0].plot(grid, np.maximum(b["median"], 1e-17), color=c, label=LABELS[name])
        s = convergence_bands(h, grid, "size_mean")
        axes[1].fill_between(grid, s["min"], s["max"], color=c, alpha=0.12, lw=0)
        axes[1].plot(grid, s["median"], color=c, label=LABELS[name])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("train NRMSE лучшей особи (значения < 1e-17 на уровне 1e-17)")
    axes[0].set_title("Сходимость: медиана и полоса мин–макс, 20 запусков", fontsize=10)
    axes[1].set_ylabel("средний размер дерева в популяции, узлов")
    axes[1].set_title("Разрастание деревьев (bloat)", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("число вызовов фитнес-функции")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=130)
    plt.close(fig)


def pooled_front(fr):
    # недоминируемые по (размер, train) точки объединённого архива; улучшение < 1e-12 считаем шумом
    fr = fr.sort_values(["size", "train"])
    rows, best = [], np.inf
    for _, r in fr.groupby("size", sort=True).head(1).iterrows():
        if r["train"] < best - 1e-12:
            rows.append(r)
            best = r["train"]
    return pd.DataFrame(rows)


def plot_fronts(fronts, out):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (name, fr), c in zip(fronts.items(), COLORS):
        if name == "random_search":
            continue
        pf = pooled_front(fr)
        ax.plot(pf["size"], np.maximum(pf["train"], 1e-17), "o-", ms=3, color=c,
                label=f"{LABELS[name]}: train")
        ax.plot(pf["size"], np.maximum(pf["test"], 1e-17), "x--", ms=4, color=c, lw=0.8,
                label=f"{LABELS[name]}: test тех же формул")
    bl = pd.read_csv(out / "baselines.csv")
    for (_, r), m in zip(bl.iterrows(), "sDP^"):
        ax.plot(r["params"], r["test"], m, color="k", ms=8, label=f"{r['model']}: test")
    ax.set_yscale("log")
    ax.set_xlabel("размер модели: узлов дерева / параметров")
    ax.set_ylabel("NRMSE (значения < 1e-17 на уровне 1e-17)")
    ax.set_title("Фронт «ошибка на train — размер» по объединённым архивам 20 запусков", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "complexity_front.png", dpi=130)
    plt.close(fig)


def write_fronts(fronts, out):
    bl = pd.read_csv(out / "baselines.csv").set_index("model")
    picks = [("простая: train лучше полинома 2-й степени", bl.loc["полином 2-й степени", "train"]),
             ("компромисс: train < 0.01", 0.01),
             ("точная: train < 1e-6", 1e-6)]
    lines = []
    for name in ["alpha1e-3", "alpha0", "alpha1e-2"]:
        if name not in fronts:
            continue
        pf = pooled_front(fronts[name])
        lines += [f"### {name}", "",
                  "| размер | train NRMSE | test NRMSE | экстраполяция NRMSE | формула |",
                  "|---|---|---|---|---|"]
        lines += [f"| {r['size']} | {r['train']:.2e} | {r['test']:.2e} | "
                  f"{r['extrapolation']:.2e} | `{r['expr']}` |" for _, r in pf.iterrows()]
        lines.append("")
        if name == "alpha1e-3":
            sel = ["| решение | размер | train | test | экстраполяция | формула |",
                   "|---|---|---|---|---|---|"]
            for label, thr in picks:
                ok = pf[pf["train"] < thr]
                if len(ok):
                    r = ok.iloc[0]
                    sel.append(f"| {label} | {r['size']} | {r['train']:.2e} | {r['test']:.2e} | "
                               f"{r['extrapolation']:.2e} | `{r['expr']}` |")
            (out / "selected.md").write_text("\n".join(sel) + "\n", encoding="utf-8")
    (out / "fronts.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--out", default="lab3/results")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--plots-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.plots_only:
        run_configs(a.config, parse_seeds(a.seeds), out, a.jobs)
    write_baselines(out)
    runs, hist, fronts = load_all(out)
    write_summary(runs, out)
    write_comparisons(runs, out)
    plot_convergence(hist, out)
    plot_fronts(fronts, out)
    write_fronts(fronts, out)
    names = [k for k in ORDER if k in hist]
    plot_box(runs, "test", names, out / "boxplot_test.png",
             labels=[LABELS[k] for k in names], floor=1e-17,
             title="test NRMSE лучшей особи, 20 запусков", ylabel="test NRMSE (лог. шкала)",
             hline=0.01, hline_label="порог «решено» 0.01")
    plot_box(runs, "size", names, out / "boxplot_size.png", logy=False,
             labels=[LABELS[k] for k in names],
             title="Размер итогового дерева, 20 запусков", ylabel="узлов")
    print(f"сводка: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
