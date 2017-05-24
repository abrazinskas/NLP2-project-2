# this file contains support functions that are specific for CRF model
from lib.libitg import CFG
from toposort import toposort
import numpy as np
from misc.log import Log
import time
import pickle
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def viterbi_decoding(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns max I(v) subtree under v potentials and back pointers """
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


def traverse_back_pointers(back_pointers, start_node, terminals_decoration_func=lambda x :x):
    """
    Recursive traversal of tree by following back-pointers. This logic is used for decoding.
    Returns a list of terminals in order.
    :param terminals_decoration_func: a function that can be used to decorate terminals, e.g. strip spans
    """
    terminals = []
    rule = back_pointers[start_node]
    for node in rule._rhs:
        __traverse(back_pointers, node, terminals, terminals_decoration_func)
    return terminals


def __traverse(back_pointers, start_node, collector, terminals_decoration_func):
    if start_node.is_terminal():
        decorated_terminal = terminals_decoration_func(start_node)
        collector.append(decorated_terminal)
        return
    rule = back_pointers[start_node]
    for node in rule._rhs:
        __traverse(back_pointers, node, collector, terminals_decoration_func)


def compute_learning_rate(start_learning_rate, step, decay_rate=1 ):
    """
    computes adaptive learning rate that is based on the number of updates already performed.
    """
    return start_learning_rate/(1.+start_learning_rate*decay_rate*step)


def evaluate(crf, featurizer, val_data_path):
    all_refs = []
    hypotheses = []
    total_loglikelihood = 0.
    counter = 0.
    for chinese, references, Dx, Dxys in pickle.load(open(val_data_path, 'rb')):
        for reference, Dxy in zip(references, Dxys):
            # there is no way we can compute log-likelihood if Dxy is empty
            if len(Dxy._rules) == 0:
                continue
            counter += 1
            crf.features = featurizer.featurize_parse_trees(Dx, Dxy, chinese)
            # compute log-likelihood
            total_loglikelihood += crf.compute_loglikelihood(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
        crf.features = featurizer.featurize_parse_trees(Dx, None, chinese)
        viterbi_y = crf.decode_viterbi(source_sentence=chinese, Dnx=Dx)
        if len(viterbi_y) > 1: # it's 1 because of /usr/local/lib/python3.6/site-packages/nltk/translate/bleu_score.py", line 544
            cur_refs = [r.split() for r in references]
            hypotheses.append(viterbi_y)
            all_refs.append(cur_refs)
    bleu = corpus_bleu(all_refs, hypotheses, smoothing_function=SmoothingFunction().method7)
    if counter == 0:
        total_loglikelihood = 0.
    else:
        total_loglikelihood /= float(counter)
    return bleu, total_loglikelihood



