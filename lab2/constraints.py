import numpy as np

from .knapsack import Instance


def penalty_rates(inst: Instance) -> tuple[float, float]:
    return float(np.max(inst.value / inst.mass)), float(np.max(inst.value / inst.volume))


def penalized_fitness(inst: Instance, X: np.ndarray, lam: float) -> np.ndarray:
    rho_m, rho_v = penalty_rates(inst)
    em, ev = inst.excess(X)
    _, _, val = inst.loads(X)
    return val - lam * (rho_m * em + rho_v * ev)


class GreedyRepair:
    def __init__(self, inst: Instance):
        self.inst = inst
        eff = inst.efficiency()
        self.worst_first = np.argsort(eff, kind="stable")
        self.best_first = self.worst_first[::-1]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        inst = self.inst
        X = np.array(X, dtype=np.int8, copy=True)
        W, V = inst.cap_mass, inst.cap_volume
        for row in X:
            m = int(row @ inst.mass)
            v = int(row @ inst.volume)
            if m > W or v > V:
                for i in self.worst_first:
                    if row[i]:
                        row[i] = 0
                        m -= inst.mass[i]
                        v -= inst.volume[i]
                        if m <= W and v <= V:
                            break
            for i in self.best_first:
                if not row[i] and m + inst.mass[i] <= W and v + inst.volume[i] <= V:
                    row[i] = 1
                    m += inst.mass[i]
                    v += inst.volume[i]
        return X
