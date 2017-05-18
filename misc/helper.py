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

# Returns a dictionary containing the IBM 1 lexical probabilities in both directions.
# probs[(chinese, english)] returns the probability of the chinese word translating
# into the english word and vice versa.
def load_ibm1_probs(ibm1_file):
    probs = defaultdict(float)
    with open(ibm1_file) as f:
        for line in f:
            chinese, english, p_e_given_f, p_f_given_e = line.split()

            if chinese == "<NULL>":
                chinese = "-EPS-"

            if english == "<NULL>":
                english = "-EPS-"

            p_e_given_f = 0.0 if p_e_given_f == "NA" else float(p_e_given_f)
            p_f_given_e = 0.0 if p_f_given_e == "NA" else float(p_f_given_e)
            probs[(chinese, english)] = p_e_given_f
            probs[(english, chinese)] = p_f_given_e
    return probs

def scan_line(line):
    split_line = line.split(" ||| ")
    chinese = split_line[0]
    english = split_line[1][:-1]
    return (chinese, english)

def featurize_edges(forest, src_fsa, ibm1_probs, eps=Terminal('-EPS-')):
    edge2fmap = dict()
    for edge in forest:
        edge2fmap[edge] = featurize_edge(edge, src_fsa, ibm1_probs, eps=eps)
    return edge2fmap
