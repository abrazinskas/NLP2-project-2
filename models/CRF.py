import numpy as np
from collections import defaultdict
from lib.libitg import Symbol, Terminal, Nonterminal, Span, Rule, FSA

from .spans import get_bispans, get_target_word, get_source_word

def featurize_edges(forest, src_fsa, ibm1_probs, eps=Terminal('-EPS-')):
    edge2fmap = dict()
    for edge in forest:
        edge2fmap[edge] = featurize_edge(edge, src_fsa, ibm1_probs, eps=eps)
    return edge2fmap

def featurize_edge(edge, src_fsa, ibm1_probs, eps=Terminal('-EPS-'), bispans=False):
    """
    Featurises an edge given
        * rule and spans
        * src sentence as an FSA
        * TODO: target sentence length n
    crucially, note that the target sentence y is not available!
    """
    fmap = defaultdict(float)

    # Check if the edge represents a binary or unary rule.
    if len(edge.rhs) == 2:
        featurize_binary_rule(edge, src_fsa, fmap, bispans)
    else:
        symbol = edge.rhs[0]

        # Check if the rule is a terminal rule or a start rule.
        if symbol.is_terminal():
            featurize_terminal_rule(edge, src_fsa, fmap, ibm1_probs, eps, bispans)
        elif edge.lhs.obj()[0] != Nonterminal("X"):
            featurize_start_rule(edge, src_fsa, fmap)
        else:
            featurize_upgrade_rule(edge, src_fsa, fmap)

    return fmap

def featurize_upgrade_rule(rule, src_fsa, fmap):
    lhs_symbol, lhs_start, lhs_end = rule.lhs.obj()
    rhs_symbol, rhs_start, rhs_end = rule.rhs[0].obj() # TODO spans prob not needed

    if lhs_symbol == Nonterminal("X"):
        if rhs_symbol == Nonterminal("T"):
            fmap["type:upgrade_t"] += 1.0
        elif rhs_symbol == Nonterminal("D"):
            fmap["type:upgrade_d"] += 1.0
        elif rhs_symbol == Nonterminal("T"):
            fmap["type:upgrade_t"] += 1.0

def featurize_binary_rule(rule, src_fsa, fmap, bispans):
    fmap['type:binary'] += 1.0

    # here we could have sparse features of the source string as a function of spans being concatenated
    if bispans:
        (ls1, ls2), (lt1, lt2) = get_bispans(rule.rhs[0])  # left of RHS
        (rs1, rs2), (rt1, rt2) = get_bispans(rule.rhs[1])  # right of RHS

        if ls1 == ls2:
            fmap['binary:deletion_left_src_child'] += 1.0
        if rs1 == rs2:
            fmap['binary:deletion_right_src_child'] += 1.0
        if ls2 == rs1:
            fmap['binary:monotone'] += 1.0
        if ls1 == rs2:
            fmap['binary:inverted'] += 1.0
    else:
        lhs_symbol, lhs_start, lhs_end = rule.lhs.obj()
        rhs_symbol_1, rhs_start_1, rhs_end_1 = rule.rhs[0].obj()
        rhs_symbol_2, rhs_start_2, rhs_end_2 = rule.rhs[1].obj()

        if lhs_symbol == Nonterminal("D"):
            fmap["binary:recursive_deletion"] += 1.0
        elif lhs_symbol == Nonterminal("X"):

            # Check for invertion.
            if lhs_start == rhs_start_2:
                fmap["binary:inverted"] += 1.0
            else:
                fmap["binary:monotone"] += 1.0

        # TODO sparser features

def featurize_start_rule(rule, src_fsa, fmap):
    fmap['top'] += 1.0

def featurize_terminal_rule(rule, src_fsa, fmap, ibm1_probs, eps, bispans):
    fmap["type:terminal"] += 1.0

    lhs_symbol, lhs_start, lhs_end = rule.lhs.obj()
    rhs_symbol, rhs_start, rhs_end = rule.rhs[0].obj() # TODO spans prob not needed

    if lhs_symbol == Nonterminal("D"):
        # Deletion of a source word.
        fmap["type:deletion"] += 1.0

        # IBM 1 deletion probabilities.
        src_word = get_source_word(src_fsa, lhs_start, lhs_end)
        fmap["ibm1:del:logprob"] += np.log(ibm1_probs[(src_word, "-EPS-")] + 1e-10)

        # Sparse deletion feature for specific words.
        fmap["del:%s" % src_word] += 1.0

    elif lhs_symbol == Nonterminal("I"):
        # Insertion of a target word.
        tgt_word = get_target_word(rhs_symbol)

        fmap["type:insertion"] += 1.0

        # IBM 1 insertion probability.
        fmap["ibm1:ins:logprob"] += np.log(ibm1_probs[("-EPS-", tgt_word)] + 1e-10)

        # Sparse insertion feature for specific target words.
        fmap["ins:%s" % tgt_word] += 1.0

    elif lhs_symbol == Nonterminal("T"):
        # Translation of a source word into a target word.
        fmap["type:translation"] += 1.0
        src_word = get_source_word(src_fsa, lhs_start, lhs_end)
        tgt_word = get_target_word(rhs_symbol)

        # IBM 1 translation probabilities.
        fmap["ibm1:x2y:logprob"] += np.log(ibm1_probs[(src_word, tgt_word)] + 1e-10)
        fmap["ibm1:y2x:logprob"] += np.log(ibm1_probs[(tgt_word, src_word)] + 1e-10)
        fmap["ibm1:geometric:log"] += np.log(np.sqrt(ibm1_probs[(src_word, tgt_word)] * \
            ibm1_probs[(tgt_word, src_word)] + 1e-10) + 1e-10)

        # Sparse word translation features.
        fmap["trans:%s/%s" % (src_word, tgt_word)] += 1.0
