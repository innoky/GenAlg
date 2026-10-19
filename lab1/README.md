# ЛР1 — функция сферы, d = 6

Вещественный ГА: турнир k = 3 или ранговая селекция, BLX-0.5, гауссова мутация
(фиксированная или самоадаптивная σ), отражение от границ, элитизм, бюджет 10⁵ вызовов.

## Запуск

Из корня репозитория:

```bash
make venv                     # один раз
make lab1                     # все конфигурации, seeds 0..19 -> lab1/results/

# или вручную:
.venv/bin/python -m lab1.run_experiments --config lab1/configs/*.json --seeds 0-19 --out lab1/results
.venv/bin/python -m lab1.run_experiments --config lab1/configs/base_fixed.json --seeds 0,1,2 --out /tmp/try
.venv/bin/python -m lab1.run_experiments --plots-only --out lab1/results   # графики/сводка из готовых CSV
```

## Конфигурации (`configs/`)

| файл | что меняется относительно `base_fixed` |
|---|---|
| `base_fixed.json` | база: μ = 50, σ = 2 фикс., p_m = 1/d |
| `selfadaptive_sigma.json` | σ самоадаптивная (логнормальная) |
| `fixed_pm1.json` | p_m = 1 |
| `selfadaptive_pm1.json` | σ самоадаптивная и p_m = 1 |
| `pop30.json`, `pop100.json` | μ = 30 / 100 |
| `ranking.json` | линейная ранговая селекция, s = 1.5 |
| `random_search.json` | случайный поиск, тот же бюджет |

## Результаты (`results/`)

- `<config>/runs.csv` — по одной строке на запуск: seed, best_f, evals, time_s,
  generations, hit_eval / hit_gen (первое f < 1e-8), best_x, final_sigma_median;
- `<config>/history.csv` — по поколениям: evals, best_so_far, pop_best/mean/median,
  diversity, sigma_median, sigma_best;
- `<config>/config.json` — копия конфигурации;
- `summary.csv` / `summary.md`, `theory.md`;
- графики: `convergence_*.png`, `boxplot_final.png`, `sigma_adaptation.png`,
  `random_search_theory.png`.

## Код

- `functions.py` — вариант и сфера;
- `operators.py` — инициализация, BLX-α, отражение/клиппинг, гауссова мутация,
  логнормальная σ;
- `ga.py` — сборка ГА по конфигурации;
- `random_search.py`;
- `run_experiments.py` — CLI.
