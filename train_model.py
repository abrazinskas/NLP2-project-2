import numpy as np
import os, sys
import lib.libitg as libitg
from lib.libitg import Symbol, Terminal, Nonterminal, Span
from lib.libitg import Rule, CFG
from lib.libitg import FSA
from misc.helper import load_lexicon, scan_line, featurize_edges, load_ibm1_probs, log_info, load_parse_trees

if len(sys.argv) < 4:
    print("Use: python parse_training_data.py <parse_tree_dir> <lexicon> <ibm1_probs>")
    sys.exit()

# Data files.
parse_tree_dir = sys.argv[1]
lexicon_file = sys.argv[2]
ibm1_probs_file = sys.argv[3]

# Load the IBM 1 probabilities and the lexicon.
log_info("Loading IBM 1 probs...")
ibm1_probs = load_ibm1_probs(ibm1_probs_file)
log_info("Loading lexicon...")
lexicon = load_lexicon(lexicon_file)

# Start training.
log_info("Starting training.")
for Dix, Dixy, chinese, english in load_parse_trees(parse_tree_dir):

    # Create the source FSA.
    src_fsa = libitg.make_fsa(chinese)

    # Featurize Dixy.
    features_Dixy = featurize_edges(Dixy, src_fsa, ibm1_probs)

log_info("Done training.")
