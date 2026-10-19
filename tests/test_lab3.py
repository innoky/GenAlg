import json
from pathlib import Path

import numpy as np

from lab3 import baselines, gp
from lab3 import operators as ops
from lab3.tree import Generator, depth, evaluate, subtree_end, to_str

CFG = Path(__file__).parents[1] / "lab3" / "configs"
X = np.array([[1.0, 2.0], [-3.0, 0.5], [0.0, 0.0]])


def valid(tree):
    return len(tree) > 0 and subtree_end(tree, 0) == len(tree)


def test_structure():
    t = ["add", "mul", "x0", "x0", "sin", "x1"]
    assert subtree_end(t, 0) == 6 and subtree_end(t, 1) == 4 and subtree_end(t, 4) == 6
    assert depth(t) == 3 and depth(["x0"]) == 1


def test_evaluate():
    t = ["add", "mul", "x0", "x0", "sin", "x1"]
    assert np.allclose(evaluate(t, X), X[:, 0] ** 2 + np.sin(X[:, 1]))
    assert np.allclose(evaluate(["div", "x0", "x1"], X), [0.5, -6.0, 1.0])  # защищённое деление
    assert np.allclose(evaluate([2.5], X), 2.5)


def test_to_str_folds_constants():
    assert to_str(["add", "x0", "mul", 2.0, 3.0]) == "x0 + 6"
    assert to_str(["sin", "x1"]) == "sin(x1)"


def test_operators_keep_trees_valid():
    rng = np.random.default_rng(0)
    gen = Generator(2)
    pop = gen.ramped(rng, 200, 2, 6)
    assert all(valid(t) and depth(t) <= 6 for t in pop)
    for _ in range(3000):
        a, b = pop[rng.integers(200)], pop[rng.integers(200)]
        c1, c2 = ops.subtree_crossover(rng, a, b, 8)
        m1 = ops.subtree_mutation(rng, c1, gen, 8)
        m2 = ops.point_mutation(rng, c2, gen)
        for t in (c1, c2, m1, m2):
            assert valid(t) and depth(t) <= 8


def test_crossover_does_not_mutate_parents():
    rng = np.random.default_rng(1)
    a, b = ["add", "x0", "x1"], ["mul", "x1", 1.5]
    ops.subtree_crossover(rng, a, b, 8)
    assert a == ["add", "x0", "x1"] and b == ["mul", "x1", 1.5]


def test_archive_front_is_nondominated():
    Xt, y = X, X[:, 0] ** 2
    obj = gp.Objective(Xt, y, 0.0)
    obj(np.array([["x0"], ["mul", "x0", "x0"], ["add", "x0", "x1"],
                  ["add", "mul", "x0", "x0", 0.0]], dtype=object))
    front = obj.front()
    sizes = [s for s, _, _ in front]
    errs = [e for _, e, _ in front]
    assert sizes == sorted(sizes) and all(np.diff(errs) < 0)
    assert front[-1][0] == 3 and front[-1][1] < 1e-12


def test_baseline_poly_exact_on_quadratic():
    rng = np.random.default_rng(0)
    Xq = rng.uniform(-1, 1, (50, 2))
    y = 1 + Xq[:, 0] - 2 * Xq[:, 0] * Xq[:, 1]
    w = baselines.fit_poly(Xq, y, 2)
    assert np.allclose(baselines.features(Xq, 2) @ w, y)


def test_gp_deterministic_and_budget():
    cfg = json.loads((CFG / "alpha1e-3.json").read_text(encoding="utf-8"))
    cfg.update(budget=1500, pop_size=100)
    r1, h1, f1 = gp.run_one(cfg, 3)
    r2, h2, f2 = gp.run_one(cfg, 3)
    assert {**r1, "time_s": 0} == {**r2, "time_s": 0} and f1 == f2
    assert r1["evals"] == 1500 and h1[-1]["evals"] == 1500
