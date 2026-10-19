import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eaga.rng import make_rng

from .constraints import penalized_fitness, penalty_rates
from .knapsack import Instance
from .operators import two_point_crossover


def bits(s: str) -> np.ndarray:
    return np.array([int(c) for c in s], dtype=np.int8)


def describe(inst: Instance, x: np.ndarray, title: str, origin: str) -> str:
    m, v, p = (int(a[0]) for a in inst.loads(x))
    em, ev = max(0, m - inst.cap_mass), max(0, v - inst.cap_volume)
    rho_m, rho_v = penalty_rates(inst)
    ok = em == 0 and ev == 0
    ids = ", ".join(map(str, np.flatnonzero(x)))
    lines = [f"{title} — {origin}", "",
             f"- предметы ({int(x.sum())} шт.): {ids}",
             f"- масса: {m} {'≤' if m <= inst.cap_mass else '>'} W = {inst.cap_mass} → "
             f"превышение {em}",
             f"- объём: {v} {'≤' if v <= inst.cap_volume else '>'} V = {inst.cap_volume} → "
             f"превышение {ev}",
             f"- ценность P = {p}; {'допустимо' if ok else 'НЕДОПУСТИМО'}"]
    if not ok:
        for lam in (0.1, 1.0):
            pen = lam * (rho_m * em + rho_v * ev)
            F = float(penalized_fitness(inst, x, lam)[0])
            lines.append(f"- штраф при λ={lam:g}: {lam:g}·({rho_m:.4f}·{em} + {rho_v:.4f}·{ev})"
                         f" = {pen:.2f}; F = {p} − {pen:.2f} = {F:.2f}")
    return "\n".join(lines) + "\n"


def examples(inst: Instance, out: Path, b: dict) -> None:
    x_opt, x_gr = bits(b["opt_x"]), bits(b["greedy_x"])
    parts = ["## Примеры решений (вычислено `lab2/report_tables.py`)\n",
             describe(inst, x_gr, "Допустимое 1", "жадная эвристика"),
             describe(inst, x_opt, "Допустимое 2", "точный оптимум DP")]
    # мутация 0->1 у оптимума
    free = np.flatnonzero(x_opt == 0)
    j = int(free[np.argmin(inst.mass[free] + inst.volume[free])])
    x_mut = x_opt.copy()
    x_mut[j] = 1
    parts.append(describe(inst, x_mut, "Недопустимое 1",
                          f"оптимум DP после инверсии бита {j} (0→1) мутацией"))
    # кроссовер двух допустимых родителей
    rng = make_rng(0)
    for t in range(1000):
        c1, c2 = two_point_crossover(rng, x_gr[None], x_opt[None])
        bad = [c for c in (c1[0], c2[0]) if not inst.feasible(c)[0]]
        if bad:
            parts.append(describe(inst, bad[0], "Недопустимое 2",
                                  "потомок двухточечного кроссовера родителей «Допустимое 1» и "
                                  f"«Допустимое 2» (rng seed 0, попытка {t + 1})"))
            break
    # финальный лидер штрафного ГА
    r = pd.read_csv(out / "pen1_uniform" / "runs.csv", dtype={"final_best_x": str})
    bad = r[~r["final_best_feasible"]]
    if len(bad):
        row = bad.iloc[0]
        parts.append(describe(inst, bits(row.final_best_x), "Недопустимое 3",
                              f"лучшая по штрафованной фитнес-функции особь финальной "
                              f"популяции, pen1_uniform, seed {row.seed} "
                              f"(таких запусков {len(bad)}/{len(r)})"))
    (out / "examples.md").write_text("\n".join(parts), encoding="utf-8")


def best_solution(inst: Instance, out: Path, b: dict) -> None:
    frames = []
    for d in sorted(p for p in out.iterdir() if (p / "runs.csv").exists()):
        frames.append(pd.read_csv(d / "runs.csv", dtype={"best_x": str}))
    runs = pd.concat(frames, ignore_index=True)
    ga_runs = runs[runs.config != "random_feasible"].dropna(subset=["best_value"])
    best = ga_runs.sort_values(["best_value", "opt_hit_eval"],
                               ascending=[False, True]).iloc[0]
    x = bits(best.best_x)
    idx = np.flatnonzero(x)
    rows = ["| id | масса | объём | ценность | p/(w/W+v/V) |", "|---|---|---|---|---|"]
    eff = inst.efficiency()
    for i in idx:
        rows.append(f"| {i} | {inst.mass[i]} | {inst.volume[i]} | {inst.value[i]} | {eff[i]:.1f} |")
    m, v, p = (int(a[0]) for a in inst.loads(x))
    rows.append(f"| итого ({len(idx)} шт.) | {m} | {v} | {p} | |")
    rows.append(f"| ёмкость | {inst.cap_mass} | {inst.cap_volume} | | |")
    rows.append(f"| заполнение | {100 * m / inst.cap_mass:.1f}% | "
                f"{100 * v / inst.cap_volume:.1f}% | | |")
    same = bool((x == bits(b["opt_x"])).all())
    cfg = json.loads((out / best.config / "config.json").read_text(encoding="utf-8"))
    in_init = pd.notna(best.opt_hit_eval) and best.opt_hit_eval <= cfg["pop_size"]
    head = (f"## Лучшее решение ГА\n\nКонфигурация `{best.config}`, seed {best.seed}; "
            f"найдено на вызове {int(best.opt_hit_eval) if pd.notna(best.opt_hit_eval) else '—'}"
            f" из {int(best.evals)}{' (в начальной популяции, после ремонта)' if in_init else ''}."
            f" Ценность {p} при оптимуме DP {b['opt_value']} "
            f"(gap {(b['opt_value'] - p) / b['opt_value'] * 100:.2f}%). "
            f"Совпадает с решением DP: {'да' if same else 'нет (альтернативный оптимум)' if p == b['opt_value'] else 'нет'}.\n\n")
    (out / "best_solution.md").write_text(head + "\n".join(rows) + "\n", encoding="utf-8")


def instance_table(inst: Instance, out: Path) -> None:
    eff = inst.efficiency()
    rows = ["| id | масса | объём | ценность | p/(w/W+v/V) |", "|---|---|---|---|---|"]
    rows += [f"| {i} | {inst.mass[i]} | {inst.volume[i]} | {inst.value[i]} | {eff[i]:.1f} |"
             for i in range(inst.n)]
    rows.append(f"| Σ | {inst.mass.sum()} | {inst.volume.sum()} | {inst.value.sum()} | |")
    corr = (f"\nКорреляция Пирсона: p~w = {np.corrcoef(inst.value, inst.mass)[0, 1]:.3f}, "
            f"p~v = {np.corrcoef(inst.value, inst.volume)[0, 1]:.3f}, "
            f"p~(w+v) = {np.corrcoef(inst.value, inst.mass + inst.volume)[0, 1]:.3f}. "
            f"W = {inst.cap_mass} ({100 * inst.cap_mass / inst.mass.sum():.1f}% Σw), "
            f"V = {inst.cap_volume} ({100 * inst.cap_volume / inst.volume.sum():.1f}% Σv).\n")
    (out / "instance.md").write_text("\n".join(rows) + "\n" + corr, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lab2/results")
    ap.add_argument("--instance", default="lab2/data/instance.json")
    a = ap.parse_args()
    out = Path(a.out)
    inst = Instance.from_json(a.instance)
    b = json.loads((out / "benchmarks.json").read_text(encoding="utf-8"))
    examples(inst, out, b)
    best_solution(inst, out, b)
    instance_table(inst, out)
    print(f"-> {out}/examples.md, best_solution.md, instance.md")


if __name__ == "__main__":
    main()
