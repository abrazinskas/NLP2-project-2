import numpy as np
from collections import defaultdict
from lib.libitg import Symbol, Terminal, Nonterminal, Span, Rule, FSA

from .spans import get_bispans, get_target_word, get_source_word

def featurize_edge(edge, src_fsa, ibm1_probs, eps=Terminal('-EPS-')):
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
        featurize_binary_rule(edge, src_fsa, fmap)
    else:
        symbol = edge.rhs[0]

        # Check if the rule is a terminal rule or a start rule.
        if symbol.is_terminal():
            featurize_terminal_rule(edge, src_fsa, fmap, ibm1_probs, eps)
        else:
            featurize_start_rule(edge, src_fsa, fmap)

    return fmap

def featurize_binary_rule(rule, src_fsa, fmap):
    fmap['type:binary'] += 1.0

    # here we could have sparse features of the source string as a function of spans being concatenated
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

def featurize_start_rule(rule, src_fsa, fmap):
    fmap['top'] += 1.0

def featurize_terminal_rule(rule, src_fsa, fmap, ibm1_probs, eps):
    fmap['type:terminal'] += 1.0

    # we could have IBM1 log probs for the traslation pair or ins/del
    symbol = rule.rhs[0]
    (s1, s2), (t1, t2) = get_bispans(symbol)
    if symbol.root() == eps:
        # Deletion:
        fmap['type:deletion'] += 1.0

        # Somehting goes wrong here, the source span can be over no source words if the inserted symbol is eps...

        # IBM 1 deletion probabilities.
        # src_word = get_source_word(src_fsa, s1, s2)
        # fmap['ibm1:del:logprob'] += np.log(ibm1_probs[(src_word, "-EPS-")] + 1e-10)

        # Sparse deletion feature for specific words.
        # fmap['del:%s' % src_word] += 1.0
    else:
        tgt_word = get_target_word(symbol)
        if s1 == s2:
            # Insertion:
            fmap['type:insertion'] += 1.0

            # IBM 1 insertion probability.
            fmap['ibm1:ins:logprob'] += np.log(ibm1_probs[(eps.root(), tgt_word)] + 1e-10)

            # Sparse insertion feature for specific target words.
            fmap['ins:%s' % tgt_word] += 1.0
        else:
            # Translation:
            src_word = get_source_word(src_fsa, s1, s2)
            fmap['type:translation'] += 1.0

            # IBM 1 translation probabilities.
            fmap['ibm1:x2y:logprob'] += np.log(ibm1_probs[(src_word, tgt_word)] + 1e-10)
            fmap['ibm1:y2x:logprob'] += np.log(ibm1_probs[(tgt_word, src_word)] + 1e-10)
            fmap['ibm1:geometric:log'] += np.log(np.sqrt(ibm1_probs[(src_word, tgt_word)] * \
                    ibm1_probs[(tgt_word, src_word)] + 1e-10) + 1e-10)

            # Sparse word translation features.
            fmap['trans:%s/%s' % (src_word, tgt_word)] += 1.0