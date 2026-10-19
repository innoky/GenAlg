import numpy as np

# дерево хранится как список токенов в префиксной записи:
# str из FUNCS — функция, "x0"/"x1" — переменная, float — константа
FUNCS = {"add": 2, "sub": 2, "mul": 2, "div": 2, "sin": 1, "cos": 1}
SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def arity(tok):
    return FUNCS.get(tok, 0) if isinstance(tok, str) else 0


def subtree_end(tree, i):
    need, j = 1, i
    while need:
        need += arity(tree[j]) - 1
        j += 1
    return j


def depth(tree):
    def rec(i):
        a = arity(tree[i])
        j, d = i + 1, 0
        for _ in range(a):
            dj, j = rec(j)
            d = max(d, dj)
        return d + 1, j
    return rec(0)[0]


def pdiv(a, b):
    return np.where(np.abs(b) > 1e-6, a / np.where(b == 0, 1.0, b), 1.0)


OPS = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": pdiv,
       "sin": np.sin, "cos": np.cos}


def evaluate(tree, X):
    def rec(i):
        t = tree[i]
        if not isinstance(t, str):
            return np.full(len(X), t), i + 1
        if t[0] == "x":
            return X[:, int(t[1:])], i + 1
        a, j = rec(i + 1)
        if FUNCS[t] == 1:
            return OPS[t](a), j
        b, k = rec(j)
        return OPS[t](a, b), k
    with np.errstate(all="ignore"):
        return rec(0)[0]


def has_var(tree):
    return any(isinstance(t, str) and t[0] == "x" for t in tree)


def to_str(tree):
    def rec(i):
        t = tree[i]
        if not isinstance(t, str):
            return f"{t:.4g}", i + 1
        if t[0] == "x":
            return t, i + 1
        end = subtree_end(tree, i)
        if not has_var(tree[i:end]):  # свёртка константного поддерева
            return f"{float(evaluate(tree[i:end], np.zeros((1, 1)))[0]):.4g}", end
        a, j = rec(i + 1)
        if FUNCS[t] == 1:
            return f"{t}({a})", j
        b, k = rec(j)
        return f"({a} {SYMBOL[t]} {b})", k
    s = rec(0)[0]
    return s[1:-1] if s.startswith("(") and s.endswith(")") else s


class Generator:
    def __init__(self, n_vars, const_range=(-2.0, 2.0), p_const=1 / 3):
        self.vars = [f"x{i}" for i in range(n_vars)]
        self.funcs = list(FUNCS)
        self.const_range = const_range
        self.p_const = p_const

    def terminal(self, rng):
        if rng.random() < self.p_const:
            return float(rng.uniform(*self.const_range))
        return self.vars[rng.integers(len(self.vars))]

    def tree(self, rng, max_depth, method="grow"):
        n_f, n_t = len(self.funcs), len(self.vars) + 1
        if max_depth <= 1 or (method == "grow" and rng.random() < n_t / (n_t + n_f)):
            return [self.terminal(rng)]
        f = self.funcs[rng.integers(n_f)]
        out = [f]
        for _ in range(FUNCS[f]):
            out += self.tree(rng, max_depth - 1, method)
        return out

    def ramped(self, rng, n, min_depth, max_depth):
        pop = np.empty(n, dtype=object)
        depths = range(min_depth, max_depth + 1)
        for k in range(n):
            d = depths[k % len(depths)]
            pop[k] = self.tree(rng, d, "full" if (k // len(depths)) % 2 else "grow")
        return pop
