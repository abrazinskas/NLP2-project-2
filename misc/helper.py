from collections import defaultdict
from lib.libitg import Terminal

from .features import featurize_edge

def load_lexicon(lexicon_file, anything_into_eps=True, eps_into_anything=True):
    lexicon = defaultdict(set)
    with open(lexicon_file) as f:
        for line in f:
            line_split = line.split(" -> ")
            assert len(line_split) == 2
            source = line_split[0]
            targets = line_split[1].split()
            targets = targets + ["-EPS-"] if anything_into_eps else targets
            lexicon[source].update(targets)

    if eps_into_anything:
        english_types = set()
        for translations in lexicon.values():
            english_types |= translations
        if "-EPS-" in english_types:
            english_types.remove("-EPS-")
        lexicon["-EPS-"].update(english_types)

    return lexicon

def scan_line(line):
    split_line = line.split(" ||| ")
    chinese = split_line[0]
    english = split_line[1][:-1]
    return (chinese, english)

def featurize_edges(forest, src_fsa, sparse_del=False, sparse_ins=False, \
        sparse_trans=False, eps=Terminal('-EPS-')) -> dict:
    edge2fmap = dict()
    for edge in forest:
        edge2fmap[edge] = featurize_edge(edge, src_fsa, eps, sparse_del, sparse_ins, sparse_trans)
    return edge2fmap
