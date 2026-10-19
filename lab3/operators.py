import numpy as np

from .tree import FUNCS, arity, depth, subtree_end


def pick_node(rng, tree, p_internal=0.9):
    internal = [i for i, t in enumerate(tree) if arity(t) > 0]
    if internal and rng.random() < p_internal:
        return internal[rng.integers(len(internal))]
    leaves = [i for i, t in enumerate(tree) if arity(t) == 0]
    return leaves[rng.integers(len(leaves))]


def subtree_crossover(rng, a, b, max_depth):
    i, j = pick_node(rng, a), pick_node(rng, b)
    ie, je = subtree_end(a, i), subtree_end(b, j)
    c1 = a[:i] + b[j:je] + a[ie:]
    c2 = b[:j] + a[i:ie] + b[je:]
    return (c1 if depth(c1) <= max_depth else list(a),
            c2 if depth(c2) <= max_depth else list(b))


def subtree_mutation(rng, tree, gen, max_depth, sub_depth=4):
    i = rng.integers(len(tree))
    new = tree[:i] + gen.tree(rng, sub_depth, "grow") + tree[subtree_end(tree, i):]
    return new if depth(new) <= max_depth else list(tree)


def point_mutation(rng, tree, gen, const_sd=0.3):
    new = list(tree)
    i = rng.integers(len(new))
    t = new[i]
    if not isinstance(t, str):
        new[i] = float(t + rng.normal(0, const_sd))
    elif t in FUNCS:
        same = [f for f in FUNCS if FUNCS[f] == FUNCS[t] and f != t]
        if same:
            new[i] = same[rng.integers(len(same))]
    else:
        new[i] = gen.terminal(rng)
    return new
