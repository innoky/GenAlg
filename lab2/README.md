# ЛР2 — рюкзак с двумя ограничениями (масса + объём)

Бинарный ГА: турнир k = 3, элитизм 2, равномерный или двухточечный кроссовер,
битовая мутация 1/n. Ограничения обрабатываются штрафом (λ = 0.1 / 1) или жадным
ремонтом. Бюджет 2·10⁴ вызовов. Бенчмарки: точная DP, жадность, случайный
допустимый поиск.

## Запуск

Из корня репозитория:

```bash
make venv     # один раз
make lab2     # генерация экземпляра (seed 527) + все конфигурации + таблицы для отчёта

# или вручную:
.venv/bin/python -m lab2.data_gen --seed 527 --out lab2/data/instance.json
.venv/bin/python -m lab2.run_experiments --config lab2/configs/*.json --seeds 0-19 --out lab2/results
.venv/bin/python -m lab2.report_tables --out lab2/results
```

## Конфигурации (`configs/`)

`{pen01, pen1, repair}_{uniform, two_point}.json`: штраф λ = 0.1, штраф λ = 1 или
ремонт × равномерный или двухточечный кроссовер. `random_feasible.json` — случайный
допустимый поиск.

## Результаты (`results/`)

- `<config>/runs.csv` — по запуску:
  - best_value — лучшая допустимая ценность (пусто, если допустимых не было);
  - final_best_feasible / final_best_value / final_best_x — лучшая по фитнесу
    особь финальной популяции;
  - доли допустимых;
  - opt_hit_eval — вызов, на котором впервые найден оптимум;
  - best_x — генотип.
- `<config>/history.csv` — по поколениям, включая best_feasible_so_far и
  pop_feasible_frac.
- `benchmarks.json` — DP-оптимум и жадность.
- `summary.csv` / `summary.md`, `comparisons.md` (перестановочные тесты).
- `examples.md`, `best_solution.md`, `instance.md` — таблицы для отчёта.
- Графики: `convergence.png`, `boxplot_final.png`, `feasibility.png`.

## Код

- `knapsack.py` — модель;
- `data_gen.py` — генератор;
- `exact_dp.py` — DP O(n·W·V);
- `constraints.py` — штраф и жадный ремонт;
- `operators.py` — кроссоверы и мутация;
- `ga.py` — ГА и фитнес;
- `baselines.py` — жадность и случайный допустимый поиск;
- `run_experiments.py` — CLI;
- `report_tables.py` — таблицы для отчёта.
