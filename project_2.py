import numpy as np
import lib.libitg as libitg
from lib.libitg import Symbol, Terminal, Nonterminal, Span
from lib.libitg import Rule, CFG
from lib.libitg import FSA
from misc.helper import load_lexicon, scan_line, featurize_edges, load_ibm1_probs

# Data files.
LEXICON = "data/top_5"
IBM1_PROBS = "data/lexicon"
TRAINING_DATA = "data/data/training.zh-en"

# Hyperparameters.
max_insertions = 4
max_length = None

# Load the IBM 1 probabilities, the lexicon and create the source CFG.
ibm1_probs = load_ibm1_probs(IBM1_PROBS)
lexicon = load_lexicon(LEXICON, anything_into_eps=True, eps_into_anything=True)
src_cfg = libitg.make_source_side_itg(lexicon)

# Training.
with open(TRAINING_DATA) as f:
    for line in f:
        chinese, english = scan_line(line)

        # Skip training sentences longer than some max_length if max_length is set.
        if max_length is not None and len(english) > max_length:
            continue

        # Create an FSA for the source sentence and parse the source sentence.
        src_fsa = libitg.make_fsa(chinese)
        _Dx = libitg.earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), \
                sprime_symbol=Nonterminal("D(x)"), clean=True)

        # Create an FSA that allows for some maximum amount of insertions of -EPS-.
        # Create D_i(x) such that it has this constraint on the number of insertions.
        eps_count_fsa = libitg.InsertionConstraint(max_insertions)
        _Dix = libitg.earley(_Dx, eps_count_fsa, start_symbol=Nonterminal('D(x)'), \
                sprime_symbol=Nonterminal('D_n(x)'), eps_symbol=None, clean=True)
        Dix = libitg.make_target_side_itg(_Dix, lexicon)

        # Create a target FSA and D_i(x, y)
        tgt_fsa = libitg.make_fsa(english)
        Dixy = libitg.earley(Dix, tgt_fsa, start_symbol=Nonterminal("D_n(x)"), \
                sprime_symbol=Nonterminal('D(x,y)'), clean=True)

        # Continue in the case the parse for this training sentence is empty.
        if len(Dix) == 0 or len(Dixy) == 0:
            continue

        # Featurize Dixy.
        features_Dixy = featurize_edges(Dixy, src_fsa, ibm1_probs)

        # Break after a single training instance for now.
        for edge, fmap in features_Dixy.items():
            print(edge)
            print(fmap)
            print()
        print("Length D_i(x): %d, length D_i(x, y): %d" % (len(Dix), len(Dixy)))
        break
