import os
import dill as pickle
import lib.libitg as libitg

from collections import defaultdict
from lib.libitg import Terminal, Nonterminal
from time import strftime, localtime

def load_lexicon(lexicon_file, top_n=5):
    lexicon = defaultdict(set)
    with open(lexicon_file) as f:
        for line in f:
            line_split = line.split(" -> ")
            assert len(line_split) == 2
            source = line_split[0]
            targets = line_split[1].split()
            targets = targets[:top_n]
            targets = targets + ["-EPS-"] if source != "-EPS-" else targets
            lexicon[source].update(targets)
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

def log_info(log_string):
    time_string = strftime("%H:%M:%S", localtime())
    print("%s [INFO]: %s" % (time_string, log_string))

def load_parse_trees(parse_tree_dir):
    dx_dir = os.path.join(parse_tree_dir, "Dx")
    dxy_dir = os.path.join(parse_tree_dir, "Dxy")
    english_filename = os.path.join(parse_tree_dir, "english")
    chinese_filename = os.path.join(parse_tree_dir, "chinese")
    with open(english_filename) as en_file, open(chinese_filename) as ch_file:
        idx = 0
        for en_sentence, ch_sentence in zip(en_file, ch_file):
            Dx_file = os.path.join(dx_dir, str(idx))
            Dxy_file = os.path.join(dxy_dir, str(idx))

            # Load Dx.
            Dx = None
            with open(Dx_file, "rb") as f:
                Dx = pickle.load(f)

            # Load Dxy.
            Dxy = None
            with open(Dxy_file, "rb") as f:
                Dxy = pickle.load(f)

            # Yield (Dx, Dxy).
            idx += 1
            yield (Dx, Dxy, ch_sentence[:-1], en_sentence[:-1])

def load_dev_data(filename, lexicon, return_Dxy=False, max_Dxy=None):
    with open(filename) as f:
        for line in f:
            splits = line.split(" ||| ")
            chinese = splits[0]
            references = splits[1:]
            sub_lexicon = {k:lexicon[k] for k in chinese.split() + ["-EPS-"] if k in lexicon}
            src_cfg = libitg.make_source_side_finite_itg(sub_lexicon)

            # Create an FSA for the source sentence and parse the source sentence.
            src_fsa = libitg.make_fsa(chinese)
            _Dx = libitg.earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), \
                    sprime_symbol=Nonterminal("D(x)"), clean=True)
            Dx = libitg.make_target_side_itg(_Dx, lexicon)

            # Create a target FSA and D(x, y) for the ref translations
            if return_Dxy:
                Dxys = []
                refs = references if max_Dxy is None else references[:max_Dxy]
                for ref in refs:
                    tgt_fsa = libitg.make_fsa(ref)
                    Dxy = libitg.earley(Dx, tgt_fsa, start_symbol=Nonterminal("D(x)"), \
                            sprime_symbol=Nonterminal('D(x,y)'), clean=True)
                    if len(Dxy._rules)>0:
                        Dxys.append(Dxy)
                yield (chinese, references, Dx, Dxys)
            else:
                yield (chinese, references, Dx)
