from collections import defaultdict
from lib.libitg import Symbol, Terminal, Nonterminal, Span, Rule, FSA

from .spans import get_bispans, get_target_word, get_source_word

def featurize_edge(edge, src_fsa, eps=Terminal('-EPS-')):
    """
    Featurises an edge given
        * rule and spans
        * src sentence as an FSA
        * TODO: target sentence length n
        * TODO: extract IBM1 dense features
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
            featurize_terminal_rule(edge, src_fsa, fmap, eps)
        else:
            featurize_start_rule(edge, src_fsa, fmap)

    return fmap

def featurize_binary_rule(rule, src_fsa, fmap):
    fmap['type:binary'] += 1.0

    # here we could have sparse features of the source string as a function of spans being concatenated
    (ls1, ls2), (lt1, lt2) = get_bispans(rule.rhs[0])  # left of RHS
    (rs1, rs2), (rt1, rt2) = get_bispans(rule.rhs[1])  # right of RHS

    if ls1 == ls2:  # deletion of source left child
        pass
    if rs1 == rs2:  # deletion of source right child
        pass
    if ls2 == rs1:  # monotone
        pass
    if ls1 == rs2:  # inverted
        pass

def featurize_start_rule(rule, src_fsa, fmap):
    fmap['top'] += 1.0

def featurize_terminal_rule(rule, src_fsa, fmap, eps):
    fmap['type:terminal'] += 1.0

    # we could have IBM1 log probs for the traslation pair or ins/del
    symbol = rule.rhs[0]
    (s1, s2), (t1, t2) = get_bispans(symbol)
    if symbol.root() == eps:  # symbol.root() gives us a Terminal free of annotation
        # for sure there is a source word
        # src_word = get_source_word(src_fsa, s1, s2)
        fmap['type:deletion'] += 1.0
        # dense versions (for initial development phase)
        # TODO: use IBM1 prob
        #ff['ibm1:del:logprob'] += 
        # sparse version
        # if sparse_del:
        #     fmap['del:%s' % src_word] += 1.0
    else:
        # for sure there's a target word
        tgt_word = get_target_word(symbol)
        if s1 == s2:  # has not consumed any source word, must be an eps rule
            fmap['type:insertion'] += 1.0
            # dense version
            # TODO: use IBM1 prob
            #ff['ibm1:ins:logprob'] += 
            # sparse version
            if True:#sparse_ins:
                fmap['ins:%s' % tgt_word] += 1.0
        else:
            # for sure there's a source word
            src_word = get_source_word(src_fsa, s1, s2)
            fmap['type:translation'] += 1.0
            # dense version
            # TODO: use IBM1 prob
            #ff['ibm1:x2y:logprob'] +=
            #ff['ibm1:y2x:logprob'] +=
            # sparse version
            if True:#sparse_trans:
                fmap['trans:%s/%s' % (src_word, tgt_word)] += 1.0
