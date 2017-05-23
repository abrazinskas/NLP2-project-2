# This file contains CRF specific support functions that are independent of inner model's parameters
from lib.libitg import CFG
from toposort import toposort
import numpy as np


def inside_algorithm(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside weight of each node in log space"""
    I = {}  # inside values in log space
    for node in tsort:
        if node.is_terminal():
            I[node] = np.log(1.)
            continue
        temp_inside = 0.
        # get rules where node appears as head
        for rule in forest._rules_by_lhs[node]:
            weight = edge_weights[rule]  # factor in log space
            inner_sum = weight
            for rhs_node in rule.rhs:
                inner_sum += I[rhs_node]
            temp_inside += np.exp(inner_sum)
        I[node] = np.log(temp_inside)
    return I

def inside_algorithm_normal(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside weight of each node """
    I = {}  # inside values in log space
    for node in tsort:
        if node.is_terminal():
            I[node] = 1.
            continue
        I[node] = 0.
        # get rules where node appears as head
        for rule in forest._rules_by_lhs[node]:
            weight = np.exp(edge_weights[rule])  # factor in log space
            inner_prod = weight
            for rhs_node in rule.rhs:
                inner_prod *= I[rhs_node]
            I[node] += inner_prod
    return I


def viterbi_decoding(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns derivations that form them argmax tree """
    I = {}
    I_a = {}  # back-pointer
    for node in tsort:
        if node.is_terminal():
            I[node] = np.log(1.)
            continue
        temp_max = np.float('-inf')
        temp_arg_max = None
        # get rules where node appears as head
        for rule in forest._rules_by_lhs[node]:
            weight = edge_weights[rule]  # factor in log space
            inner_sum = weight
            for rhs_node in rule.rhs:
                inner_sum += I[rhs_node]
            if inner_sum > temp_max:
                temp_max = inner_sum
                temp_arg_max = rule
        I[node] = temp_max
        I_a[node] = temp_arg_max
    return I, I_a


def outside_algorithm(forest: CFG, tsort:list, edge_weights: dict, inside: dict) -> dict:
    """Returns the outside weight of each node in log space"""
    O = {}  # outside values
    for i, node in enumerate(reversed(tsort)):
        # check if the node is root, assuming that it appears fist
        if i == 0:
            O[node] = np.log(1.)
            continue
        temp_outside = 0.
        # get rules where node appears as child
        for rule in forest._rules_by_rhs[node]:
            w = edge_weights[rule]
            inner_sum = O[rule._lhs] + w
            for rhs_node in rule._rhs:
                if rhs_node == node:
                    continue
                inner_sum += inside[rhs_node]
            temp_outside += np.exp(inner_sum)
        O[node] = np.log(temp_outside)
    return O


def expected_feature_vector(forest: CFG, inside: dict, outside: dict, edge_features) -> dict:
    """Returns an expected feature vector (here a sparse python dictionary)"""
    phi = {}
    for rule in forest._rules:
        k = outside[rule._lhs]
        for rhs_node in rule._rhs:
            k += inside[rhs_node]
        features = edge_features(rule)
        for feature_name, feature_value in features.items():
            if feature_name not in phi:
                phi[feature_name] = 0.
            phi[feature_name] += k * feature_value
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