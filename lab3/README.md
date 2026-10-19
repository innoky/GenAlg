# ЛР3 — символьная регрессия (трек B, вариант 1)

Генетическое программирование на деревьях выражений. Функции: + − × ÷ (защищённое),
sin, cos. Терминалы: x0, x1, случайные константы из U[−2, 2]. Фитнес:
NRMSE на train + α·(число узлов). Внешний архив хранит недоминируемые
по паре (ошибка, размер) формулы.

Целевая зависимость: y = x0² + x0·sin(x1) + 0.5·x1. Выборки:
- train и test: по 200 точек из U[−3, 3]²;
- extrapolation: 200 точек из U[−5, 5]².

## Запуск

```bash
make lab3          # данные (seed 327) + 4 конфигурации × 20 seed, параллельно
.venv/bin/python -m lab3.run_experiments --config lab3/configs/alpha1e-3.json --seeds 0-4 --jobs 4 --out /tmp/try
.venv/bin/python -m lab3.run_experiments --plots-only
```

Результат не зависит от `--jobs`: у каждого seed свой генератор.

## Конфигурации

| файл | что меняется |
|---|---|
| `alpha1e-3.json` | база, штраф 1e-3 за узел |
| `alpha0.json` | без штрафа |
| `alpha1e-2.json` | штраф 1e-2 |
| `random_search.json` | случайные деревья, тот же бюджет и фитнес |

## Результаты (`results/`)

- `<config>/runs.csv` — по запуску: ошибки на train, test и extrapolation, размер,
  глубина, формула, дерево в JSON;
- `<config>/history.csv` — по поколениям: лучший cost, train-ошибка лучшего,
  средний и максимальный размер;
- `<config>/extra.csv` — архив недоминируемых формул каждого запуска;
- `summary.md`, `comparisons.md`, `baselines.md`, `fronts.md`, `selected.md`;
- графики: `convergence.png`, `complexity_front.png`, `boxplot_test.png`,
  `boxplot_size.png`.

## Код

- `tree.py` — представление, вычисление, печать, генерация деревьев;
- `operators.py` — кроссовер и мутации поддеревьев, точечная мутация;
- `gp.py` — фитнес, архив, запуск;
- `baselines.py` — константа, линейная регрессия, полиномы (МНК);
- `data_gen.py`;
- `run_experiments.py`.
