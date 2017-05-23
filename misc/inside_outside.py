# This file contains functions for inside and outside values computation
from lib.libitg import CFG
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

