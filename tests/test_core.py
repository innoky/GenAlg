import numpy as np
import pytest

from eaga.core import BudgetExhausted, CountedFunction, run_ga
from eaga.rng import make_rng
from eaga.runner import parse_seeds, step_on_grid, summarize
from eaga.selection import linear_ranking, tournament


def sq(X):
    return np.sum(np.atleast_2d(X) ** 2, axis=1)


def test_counter_counts_rows_and_tracks_best():
    f = CountedFunction(sq, budget=10, target=0.5)
    f(np.array([[2.0], [1.0]]))
    assert f.evals == 2 and f.best_cost == 1.0 and f.first_hit_eval is None
    f(np.array([[3.0], [0.1], [0.0]]))
    assert f.evals == 5 and f.first_hit_eval == 4
    assert f.best_cost == 0.0


def test_counter_refuses_over_budget():
    f = CountedFunction(sq, budget=3)
    f(np.zeros((2, 1)))
    with pytest.raises(BudgetExhausted):
        f(np.zeros((2, 1)))
    assert f.evals == 2


def test_rng_reproducible():
    a = make_rng(7).random(5)
    b = make_rng(7).random(5)
    c = make_rng(8).random(5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_tournament_k_equals_pop_always_picks_best():
    rng = make_rng(0)
    costs = np.array([5.0, 1.0, 3.0])
    idx = tournament(rng, costs, 100, k=50)
    assert np.all(idx == 1)


def test_tournament_pressure_prefers_better():
    rng = make_rng(1)
    costs = np.arange(20.0)
    idx = tournament(rng, costs, 20000, k=3)
    counts = np.bincount(idx, minlength=20)
    assert counts[0] > counts[10] > counts[19]
    assert abs(counts[0] / 20000 - (1 - (19 / 20) ** 3)) < 0.01


def test_linear_ranking_expected_copies():
    rng = make_rng(2)
    costs = make_rng(3).permutation(10).astype(float)
    n = 10 * 2000
    idx = linear_ranking(rng, costs, n, s=1.5)
    counts = np.bincount(idx, minlength=10) / 2000
    best, worst = int(np.argmin(costs)), int(np.argmax(costs))
    assert abs(counts[best] - 1.5) < 0.05 and abs(counts[worst] - 0.5) < 0.05


def test_run_ga_uses_exact_budget_and_keeps_elite():
    f = CountedFunction(sq, budget=1003)
    rng = make_rng(0)
    best_trace = []

    def log(pop, costs):
        best_trace.append(costs.min())
        return {}

    res = run_ga(f, rng, pop_size=10, n_elite=1,
                 init=lambda r, n: r.uniform(-5, 5, (n, 2)),
                 select=tournament,
                 vary=lambda r, P: P + r.normal(0, 0.1, P.shape),
                 log_extra=log)
    assert res.evals == 1003 == f.evals
    assert np.all(np.diff(best_trace) <= 0)
    assert res.history[-1]["evals"] == 1003


def test_parse_seeds_and_summary():
    assert parse_seeds("0-3,7") == [0, 1, 2, 3, 7]
    import pandas as pd
    df = pd.DataFrame({"config": ["a"] * 3, "v": [1.0, 2.0, 6.0]})
    s = summarize(df, "v", minimize=True).iloc[0]
    assert (s.best, s.worst, s["median"], s["mean"]) == (1.0, 6.0, 2.0, 3.0)
    assert abs(s["std"] - np.std([1, 2, 6], ddof=1)) < 1e-12


def test_step_on_grid():
    out = step_on_grid(np.array([10, 20]), np.array([5.0, 3.0]), np.array([5, 10, 15, 25]))
    assert np.isnan(out[0]) and list(out[1:]) == [5.0, 5.0, 3.0]
