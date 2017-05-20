# This file contains CRF specific support functions that are independent of inner model's parameters
from lib.libitg import Rule, CFG, FSA, Terminal
from toposort import toposort
import numpy as np

EPS = 1e-5

def inside_algorithm(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside weight of each node"""
    I = {}  # inside values
    for node in tsort:
        # check if the node is terminal
        if node.is_terminal():
            I[node] = 1
            continue
        # get rules where node appears as head
        temp_I = 0
        for rule in forest._rules_by_lhs[node]:
            w = edge_weights[rule]
            # node -> X, Y
            temp_prod = np.exp(1)  # perform computations in log scale so we will sum instead of mult.
            for rhs_node in rule.rhs:
                temp_prod += np.exp(I[rhs_node])
            temp_I += w * np.log(temp_prod)
        I[node] = temp_I
    return I


def outside_algorithm(forest: CFG, tsort:list, edge_weights: dict, inside: dict) -> dict:
    """Returns the outside weight of each node"""
    O = {}  # outside values
    for i, node in enumerate(reversed(tsort)):
        # check if the node is root, assuming that it appears last
        if i == 0:
            O[node] = 1
            continue
        # get rules where node appears as child
        temp_outside = 0
        for rule in forest._rules_by_rhs[node]:
            k = O[rule._lhs] * edge_weights[rule]  # notation like in pseudocode
            temp_prod = np.exp(1.)  # perform computations in log scale so we will sum instead of mult.
            for rhs_node in rule._rhs:
                if rhs_node == node:
                    continue
                temp_prod += np.exp(inside[rhs_node])
            temp_outside += k * np.log(temp_prod + EPS)
        O[node] = temp_outside
    return O


def expected_feature_vector(forest: CFG, inside: dict, outside: dict, edge_features) -> dict:
    """Returns an expected feature vector (here a sparse python dictionary)"""
    phi = {}
    for rule in forest._rules:
        k = np.exp(outside[rule._lhs])
        for rhs_node in rule._rhs:
            k += np.exp(inside[rhs_node])
        features = edge_features(rule)
        for feature_name, feature_value in features.items():
            if feature_name not in phi:
                phi[feature_name] = 0
            phi[feature_name] += np.log(k + EPS) * feature_value
    return phi


def top_sort(forest: CFG) -> list:
    """Returns ordered list of nodes according to topsort order in an acyclic forest"""
    # the idea is to traverse each rule, by creating a dependency set that is inputted to toposort
    # we traverse by adding dependency in such a way: X->Y,Z, then X depends on Y, and Z
    dependencies = {}
    for rule in forest._rules:
        lhs = rule.lhs
        if lhs not in dependencies:
            dependencies[lhs] = set()
        for rhs in rule.rhs:
            dependencies[lhs].add(rhs)
    # run topological sort
    temp_res = list(toposort(dependencies))
    # flatten
    res = []
    for a in temp_res:
        for b in a:
            res.append(b)
    return res