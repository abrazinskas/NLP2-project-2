from libitg import Rule, CFG, FSA, Terminal
from models.feature_functions import get_bispans, get_source_word, get_target_word
from collections import defaultdict


def featurize_edges(forest, src_fsa,
                    sparse_del=False, sparse_ins=False, sparse_trans=False,
                    eps=Terminal('-EPS-')) -> dict:
    edge2fmap = dict()
    for edge in forest:
        edge2fmap[edge] = simple_features(edge, src_fsa, eps, sparse_del, sparse_ins, sparse_trans)
    return edge2fmap


def simple_features(edge: Rule, src_fsa: FSA, eps=Terminal('-EPS-'),
                    sparse_del=False, sparse_ins=False, sparse_trans=False) -> dict:
    """
    Featurises an edge given
        * rule and spans
        * src sentence as an FSA
        * TODO: target sentence length n
        * TODO: extract IBM1 dense features
    crucially, note that the target sentence y is not available!
    """
    fmap = defaultdict(float)
    if len(edge.rhs) == 2:  # binary rule
        fmap['type:binary'] += 1.0
        # here we could have sparse features of the source string as a function of spans being concatenated
        (ls1, ls2), (lt1, lt2) = get_bispans(edge.rhs[0])  # left of RHS
        (rs1, rs2), (rt1, rt2) = get_bispans(edge.rhs[1])  # right of RHS
        # TODO: double check these, assign features, add some more
        if ls1 == ls2:  # deletion of source left child
            pass
        if rs1 == rs2:  # deletion of source right child
            pass
        if ls2 == rs1:  # monotone
            pass
        if ls1 == rs2:  # inverted
            pass
    else:  # unary
        symbol = edge.rhs[0]
        if symbol.is_terminal():  # terminal rule
            fmap['type:terminal'] += 1.0
            # we could have IBM1 log probs for the traslation pair or ins/del
            (s1, s2), (t1, t2) = get_bispans(symbol)
            if symbol.root() == eps:  # symbol.root() gives us a Terminal free of annotation
                # for sure there is a source word
                src_word = get_source_word(src_fsa, s1, s2)
                fmap['type:deletion'] += 1.0
                # dense versions (for initial development phase)
                # TODO: use IBM1 prob
                #ff['ibm1:del:logprob'] +=
                # sparse version
                if sparse_del:
                    fmap['del:%s' % src_word] += 1.0
            else:
                # for sure there's a target word
                tgt_word = get_target_word(symbol)
                if s1 == s2:  # has not consumed any source word, must be an eps rule
                    fmap['type:insertion'] += 1.0
                    # dense version
                    # TODO: use IBM1 prob
                    #ff['ibm1:ins:logprob'] +=
                    # sparse version
                    if sparse_ins:
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
                    if sparse_trans:
                        fmap['trans:%s/%s' % (src_word, tgt_word)] += 1.0
        else:  # S -> X
            fmap['top'] += 1.0
    return fmap

def weight_function(edge, fmap, wmap) -> float:
    pass  # dot product of fmap and wmap  (working in log-domain)