import json
from pathlib import Path

import numpy as np

from eaga.rng import make_rng
from lab1 import ga, random_search
from lab1.functions import DIM, sphere, variant
from lab1.operators import blx_alpha, gauss_mutation, lognormal_sigma, reflect

CFG = Path(__file__).parents[1] / "lab1" / "configs"


def load(name, **over):
    cfg = json.loads((CFG / f"{name}.json").read_text(encoding="utf-8"))
    cfg.update(over)
    return cfg


def test_variant():
    assert variant(20, 2, 26) == (1, 6) and DIM == 6


def test_sphere():
    assert sphere(np.zeros((1, 6)))[0] == 0.0
    assert np.allclose(sphere(np.array([[1, 2, 3, 0, 0, 0], [1] * 6])), [14, 6])


def test_reflect_inside_unchanged_bitwise():
    x = np.array([[1e-20, -3.3e-15, 99.9, -100.0, 100.0, 0.0]])
    assert np.array_equal(reflect(x, -100, 100), x)


def test_reflect_outside():
    x = np.array([[101.0, -105.0, 350.0, -100.5]])
    y = reflect(x, -100, 100)
    assert np.allclose(y, [[99.0, -95.0, -50.0, -99.5]])
    big = make_rng(0).normal(0, 1e4, (1000, 6))
    yb = reflect(big, -100, 100)
    assert yb.min() >= -100 and yb.max() <= 100


def test_blx_range():
    rng = make_rng(0)
    A, B = np.zeros((5000, 1)), np.ones((5000, 1))
    C1, C2 = blx_alpha(rng, A, B, 0.5)
    C = np.concatenate([C1, C2])
    assert C.min() >= -0.5 and C.max() <= 1.5
    assert C.min() < -0.45 and C.max() > 1.45
    S1, _ = blx_alpha(rng, np.full((3, 2), 7.0), np.full((3, 2), 7.0), 0.5)
    assert np.all(S1 == 7.0)


def test_gauss_mutation_rate_and_scale():
    rng = make_rng(0)
    X = np.zeros((20000, 6))
    Y = gauss_mutation(rng, X, 2.0, 1 / 6)
    changed = Y != 0
    assert abs(changed.mean() - 1 / 6) < 0.01
    assert abs(Y[changed].std() - 2.0) < 0.05


def test_lognormal_sigma_bounds():
    s = lognormal_sigma(make_rng(0), np.full(1000, 1e-12), 1.0, 1e-12, 200)
    assert s.min() >= 1e-12 and s.max() <= 200


def test_ga_deterministic_and_budget():
    cfg = load("selfadaptive_pm1", budget=3000)
    r1, h1 = ga.run_one(cfg, 5)
    r2, h2 = ga.run_one(cfg, 5)
    assert r1["best_f"] == r2["best_f"] and r1["best_x"] == r2["best_x"]
    assert [h["best_so_far"] for h in h1] == [h["best_so_far"] for h in h2]
    assert r1["evals"] == 3000


def test_ga_beats_random_search_small_budget():
    g = ga.run_one(load("base_fixed", budget=5000), 0)[0]["best_f"]
    r = random_search.run_one(load("random_search", budget=5000), 0)[0]["best_f"]
    assert g < r


def test_random_search_budget_exact():
    r, h = random_search.run_one(load("random_search", budget=1234), 3)
    assert r["evals"] == 1234 and h[-1]["evals"] == 1234
