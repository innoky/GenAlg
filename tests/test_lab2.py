import itertools
import json
from pathlib import Path

import numpy as np

from eaga.rng import make_rng
from lab2 import baselines, exact_dp, ga
from lab2.constraints import GreedyRepair, penalized_fitness
from lab2.data_gen import generate
from lab2.knapsack import Instance
from lab2.operators import bitflip, two_point_crossover, uniform_crossover

CFG = Path(__file__).parents[1] / "lab2" / "configs"


def small_instance(n=12, seed=1):
    rng = make_rng(seed)
    m = rng.integers(1, 20, n)
    v = rng.integers(1, 20, n)
    p = rng.integers(1, 30, n)
    return Instance(m, v, p, int(0.4 * m.sum()), int(0.4 * v.sum()), {})


def brute_force(inst):
    best = 0
    for bits in itertools.product([0, 1], repeat=inst.n):
        x = np.array(bits)
        if inst.feasible(x)[0]:
            best = max(best, int(inst.loads(x)[2][0]))
    return best


def test_dp_matches_brute_force():
    for seed in range(3):
        inst = small_instance(seed=seed)
        opt, x, _ = exact_dp.solve(inst)
        assert opt == brute_force(inst)
        assert inst.feasible(x)[0] and inst.loads(x)[2][0] == opt


def test_generator_reproducible_and_saved_instance_matches():
    a, b = generate(527), generate(527)
    assert np.array_equal(a.value, b.value) and a.cap_mass == b.cap_mass
    saved = Instance.from_json()
    assert np.array_equal(saved.mass, a.mass) and np.array_equal(saved.value, a.value)
    assert saved.n == 40


def test_repair_always_feasible_and_maximal():
    inst = Instance.from_json()
    X = (make_rng(0).random((300, inst.n)) < 0.7).astype(np.int8)
    R = GreedyRepair(inst)(X)
    assert inst.feasible(R).all()
    m, v, _ = inst.loads(R)
    for row, mm, vv in zip(R, m, v):
        free = row == 0
        fits = (mm + inst.mass[free] <= inst.cap_mass) & (vv + inst.volume[free] <= inst.cap_volume)
        assert not fits.any()


def test_penalty_zero_iff_feasible():
    inst = Instance.from_json()
    X = (make_rng(1).random((200, inst.n)) < 0.4).astype(np.int8)
    F = penalized_fitness(inst, X, 1.0)
    val = inst.loads(X)[2]
    feas = inst.feasible(X)
    assert np.all(F[feas] == val[feas]) and np.all(F[~feas] < val[~feas])


def test_crossovers_preserve_genes_per_locus():
    rng = make_rng(0)
    A = (rng.random((50, 40)) < 0.5).astype(np.int8)
    B = (rng.random((50, 40)) < 0.5).astype(np.int8)
    for cx in (uniform_crossover, two_point_crossover):
        C1, C2 = cx(rng, A, B)
        assert np.array_equal(C1 + C2, A + B)


def test_two_point_is_contiguous_segment():
    rng = make_rng(3)
    A, B = np.zeros((200, 40), np.int8), np.ones((200, 40), np.int8)
    C1, _ = two_point_crossover(rng, A, B)
    for row in C1:
        idx = np.flatnonzero(row)
        assert idx.size == 0 or idx[-1] - idx[0] + 1 == idx.size


def test_bitflip_rate():
    X = np.zeros((5000, 40), np.int8)
    Y = bitflip(make_rng(0), X, 1 / 40)
    assert abs(Y.mean() - 1 / 40) < 0.002


def test_greedy_and_random_feasible():
    inst = Instance.from_json()
    assert inst.feasible(baselines.greedy(inst))[0]
    assert inst.feasible(baselines.random_feasible(make_rng(0), inst, 50)).all()


def test_ga_deterministic_budget_and_repair_feasible():
    cfg = json.loads((CFG / "repair_two_point.json").read_text(encoding="utf-8"))
    cfg["budget"] = 1500
    r1, h1 = ga.run_one(cfg, 4)
    r2, _ = ga.run_one(cfg, 4)
    assert r1 == {**r2, "time_s": r1["time_s"]}
    assert r1["evals"] == 1500 and r1["final_pop_feasible_frac"] == 1.0
    assert all(h["pop_feasible_frac"] == 1.0 for h in h1)
