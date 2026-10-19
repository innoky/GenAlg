import numpy as np


def features(X, degree):
    cols = [np.ones(len(X))]
    if degree >= 1:
        cols += [X[:, 0], X[:, 1]]
    if degree >= 2:
        cols += [X[:, 0] ** 2, X[:, 0] * X[:, 1], X[:, 1] ** 2]
    if degree >= 3:
        cols += [X[:, 0] ** 3, X[:, 0] ** 2 * X[:, 1], X[:, 0] * X[:, 1] ** 2, X[:, 1] ** 3]
    return np.column_stack(cols)


def fit_poly(X, y, degree):
    w, *_ = np.linalg.lstsq(features(X, degree), y, rcond=None)
    return w


def nrmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)) / np.std(y))


def evaluate(data):
    Xtr, ytr = data["train"]
    rows = []
    for name, degree in [("константа (среднее)", 0), ("линейная регрессия", 1),
                         ("полином 2-й степени", 2), ("полином 3-й степени", 3)]:
        w = fit_poly(Xtr, ytr, degree)
        row = {"model": name, "params": len(w)}
        for k, (X, y) in data.items():
            row[k] = nrmse(features(X, degree) @ w, y)
        rows.append(row)
    return rows
