import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eaga.plotting import COLORS, convergence_bands, plot_box
from eaga.runner import load_config, parse_seeds, run_series, summarize

from . import baselines, exact_dp, ga
from .knapsack import Instance

BUDGET = 20_000
LABELS = {
    "pen01_uniform": "штраф λ=0.1, равном.",
    "pen01_two_point": "штраф λ=0.1, 2-точ.",
    "pen1_uniform": "штраф λ=1, равном.",
    "pen1_two_point": "штраф λ=1, 2-точ.",
    "repair_uniform": "ремонт, равном.",
    "repair_two_point": "ремонт, 2-точ.",
    "random_feasible": "случайный допуст. поиск",
}
ORDER = list(LABELS)


def benchmarks(inst: Instance, out: Path) -> dict:
    opt, x_opt, t_dp = exact_dp.solve(inst)
    xg = baselines.greedy(inst)
    _, _, vg = inst.loads(xg)
    b = {"opt_value": opt, "dp_time_s": t_dp, "opt_x": "".join(map(str, x_opt)),
         "dp_table_cells": (inst.cap_mass + 1) * (inst.cap_volume + 1) * inst.n,
         "greedy_value": int(vg[0]), "greedy_gap_pct": (opt - int(vg[0])) / opt * 100,
         "greedy_x": "".join(map(str, xg))}
    (out / "benchmarks.json").write_text(json.dumps(b, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return b


def run_configs(paths: list[str], seeds: list[int], out: Path, inst: Instance,
                opt: int) -> None:
    for p in paths:
        cfg = load_config(p)
        fn = baselines.run_random_feasible if cfg["algorithm"] == "random_feasible" else ga.run_one
        runs, _ = run_series(cfg, seeds, lambda c, s: fn(c, s, inst=inst, opt=opt), out)
        print(f"{cfg['name']:>18}: median best = {runs['best_value'].median():.0f}, "
              f"оптимум в {int(runs['opt_hit_eval'].notna().sum())}/{len(runs)}")


def load_all(out: Path):
    runs, hist = [], {}
    for name in ORDER:
        d = out / name
        if (d / "runs.csv").exists():
            runs.append(pd.read_csv(d / "runs.csv"))
            hist[name] = pd.read_csv(d / "history.csv")
    return pd.concat(runs, ignore_index=True), hist


def write_summary(runs: pd.DataFrame, b: dict, out: Path) -> pd.DataFrame:
    opt = b["opt_value"]
    runs = runs.copy()
    runs["gap_pct"] = (opt - runs["best_value"]) / opt * 100
    runs["found_opt"] = runs["best_value"] >= opt
    s = summarize(runs, "best_value", minimize=False)
    g = runs.groupby("config", sort=False)
    extra = pd.DataFrame({
        "feasible_found": g["best_value"].count(),
        "gap_mean_pct": g["gap_pct"].mean(), "gap_median_pct": g["gap_pct"].median(),
        "gap_worst_pct": g["gap_pct"].max(), "opt_found": g["found_opt"].sum(),
        "final_best_feasible_rate": g["final_best_feasible"].mean(),
        "final_pop_feasible_mean": g["final_pop_feasible_frac"].mean(),
        "feasible_evaluated_mean": g["feasible_evaluated_frac"].mean(),
        "opt_hit_eval_median": g["opt_hit_eval"].median(),
        "time_s_mean": g["time_s"].mean(), "evals_mean": g["evals"].mean()}).reset_index()
    s = s.merge(extra, on="config")
    s.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    with open(out / "summary.md", "w", encoding="utf-8") as fh:
        fh.write(f"Оптимум (DP) = {opt}; жадная эвристика = {b['greedy_value']} "
                 f"(gap {b['greedy_gap_pct']:.2f}%).\n\n"
                 "Статистики ценности и gap считаются по запускам, в которых найдено хотя бы "
                 "одно допустимое решение (столбец «допуст. найдено»).\n\n")
        fh.write("| конфигурация | запусков | допуст. найдено | best | mean | median | std | worst | gap mean, % | "
                 "gap median, % | gap worst, % | найден оптимум | финальный лучший допустим | "
                 "доля допуст. в финальной поп. | доля допуст. среди вычисленных | "
                 "медиана вызова нахождения оптимума | вызовов | время, с |\n")
        fh.write("|" + "---|" * 18 + "\n")
        for _, r in s.iterrows():
            hit = "—" if pd.isna(r.opt_hit_eval_median) else f"{r.opt_hit_eval_median:.0f}"
            fh.write(f"| {r.config} | {r.runs} | {r.feasible_found} | {r.best:.0f} | {r['mean']:.1f} | {r['median']:.1f} | "
                     f"{r['std']:.2f} | {r.worst:.0f} | {r.gap_mean_pct:.2f} | "
                     f"{r.gap_median_pct:.2f} | {r.gap_worst_pct:.2f} | {int(r.opt_found)}/{r.runs} | "
                     f"{r.final_best_feasible_rate:.2f} | {r.final_pop_feasible_mean:.2f} | "
                     f"{r.feasible_evaluated_mean:.2f} | {hit} | {r.evals_mean:.0f} | "
                     f"{r.time_s_mean:.2f} |\n")
    return s


PAIRS = [("repair_uniform", "repair_two_point"), ("pen1_uniform", "pen1_two_point"),
         ("pen1_uniform", "repair_uniform"), ("pen1_two_point", "repair_two_point"),
         ("repair_uniform", "random_feasible")]


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 20000,
                     seed: int = 0) -> float:
    from eaga.rng import make_rng
    rng = make_rng(seed)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        hits += abs(p[:len(a)].mean() - p[len(a):].mean()) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def write_comparisons(runs: pd.DataFrame, opt: int, out: Path) -> None:
    lines = ["| A | B | mean A | mean B | найден оптимум A | найден оптимум B | p-value (перест. тест, 20000 перест.) |",
             "|---|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        va = runs.loc[runs.config == a, "best_value"].to_numpy(float)
        vb = runs.loc[runs.config == b, "best_value"].to_numpy(float)
        if len(va) == 0 or len(vb) == 0:
            continue
        lines.append(f"| {a} | {b} | {va.mean():.2f} | {vb.mean():.2f} | "
                     f"{int((va >= opt).sum())}/{len(va)} | {int((vb >= opt).sum())}/{len(vb)} | "
                     f"{permutation_test(va, vb):.4f} |")
    (out / "comparisons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_convergence(hist: dict[str, pd.DataFrame], b: dict, out: Path) -> None:
    grid = np.unique(np.geomspace(48, BUDGET, 300).round()).astype(int)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (name, h), c in zip(hist.items(), COLORS):
        if name.startswith("pen01"):
            continue
        h = h.assign(best_feasible_so_far=h["best_feasible_so_far"].where(
            h["best_feasible_so_far"] >= 0))
        bd = convergence_bands(h, grid, "best_feasible_so_far")
        ax.fill_between(grid, bd["min"], bd["max"], color=c, alpha=0.12, linewidth=0)
        ax.plot(grid, bd["median"], color=c, lw=1.6, label=LABELS[name])
        ax.plot(grid, bd["mean"], color=c, lw=0.9, ls="--")
    ax.axhline(b["opt_value"], color="k", ls=":", label=f"оптимум DP = {b['opt_value']}")
    ax.axhline(b["greedy_value"], color="gray", ls="-.", label=f"жадность = {b['greedy_value']}")
    ax.set_xscale("log")
    ax.set_ylim(b["greedy_value"] - 120, b["opt_value"] + 10)
    ax.set_xlabel("число вызовов фитнес-функции")
    ax.set_ylabel("лучшая ДОПУСТИМАЯ ценность")
    ax.set_title("Рюкзак (масса+объём), 20 запусков: медиана, полоса мин–макс, штрих — среднее\n"
                 "(штраф λ=0.1 не показан: в ≥10 из 20 запусков допустимых решений нет)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=130)
    plt.close(fig)


def plot_feasibility(hist: dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (name, h), c in zip(hist.items(), COLORS):
        if "pop_feasible_frac" not in h or name.startswith("repair"):
            continue
        m = h.groupby("gen").agg(evals=("evals", "median"), fr=("pop_feasible_frac", "mean"))
        ax.plot(m["evals"], m["fr"], color=c, label=LABELS[name])
    ax.set_xscale("log")
    ax.set_xlabel("число вызовов")
    ax.set_ylabel("доля допустимых особей в популяции")
    ax.set_title("Штрафной подход: доля допустимых в популяции (среднее по 20 запускам);\n"
                 "при ремонте она всегда 1", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "feasibility.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--out", default="lab2/results")
    ap.add_argument("--instance", default="lab2/data/instance.json")
    ap.add_argument("--plots-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inst = Instance.from_json(a.instance)
    b = benchmarks(inst, out)
    print(f"DP: оптимум {b['opt_value']} за {b['dp_time_s']:.3f} с; жадность {b['greedy_value']}")
    if not a.plots_only:
        run_configs(a.config, parse_seeds(a.seeds), out, inst, b["opt_value"])
    runs, hist = load_all(out)
    write_summary(runs, b, out)
    write_comparisons(runs, b["opt_value"], out)
    plot_convergence(hist, b, out)
    plot_feasibility(hist, out)
    names = [k for k in ORDER if k in hist]
    plot_box(runs, "best_value", names, out / "boxplot_final.png", logy=False,
             labels=[LABELS[k].replace(", ", ",\n") + "\n(допуст. "
                     f"{runs.loc[runs.config == k, 'best_value'].notna().sum()}/"
                     f"{(runs.config == k).sum()})" for k in names],
             title="Лучшая допустимая ценность после 2·10⁴ вызовов (20 запусков)",
             ylabel="ценность", hline=b["opt_value"], hline_label="оптимум DP")
    print(f"сводка: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
