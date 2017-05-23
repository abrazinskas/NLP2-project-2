import numpy as np
import lib.libitg as libitg
from lib.libitg import Symbol, Terminal, Nonterminal, Span
from lib.libitg import CFG
import os
import errno
from misc.helper import load_parse_trees



def create_folders_if_not_exist(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def get_run_var(dir):
    if not os.path.exists(dir):
        return 0
    subdirectories = get_immediate_subdirectories(dir)
    return len(subdirectories)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def create_batches(parse_tree_dir, batch_size):
    current_batch = []
    # parses : (Dx, Dxy, chinese, english)
    for i, parses in enumerate(load_parse_trees(parse_tree_dir)):
        current_batch.append(parses)
        if len(current_batch) >= batch_size:
            yield current_batch
            current_batch = []
    yield current_batch


def extend_forest_with_rules_by_rhs(forest: CFG):
    """
    Extends the forest by adding _rules_by_rhs attribute
    :param forest: CFG object
    """
    by_rhs = {}
    for rule in forest._rules:
        for rhs_node in rule._rhs:
            if rhs_node not in by_rhs:
                by_rhs[rhs_node] = []
            by_rhs[rhs_node].append(rule)
    forest._rules_by_rhs = by_rhs

