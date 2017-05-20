import numpy as np
import os, sys
import lib.libitg as libitg
from lib.libitg import Symbol, Terminal, Nonterminal, Span
from lib.libitg import Rule, CFG
from lib.libitg import FSA
from misc.helper import load_ibm1_probs, log_info, load_parse_trees
from models.CRF import featurize_edges

if len(sys.argv) < 3:
    print("Use: python parse_training_data.py <parse_tree_dir> <ibm1_probs>")
    sys.exit()

# Data files.
parse_tree_dir = sys.argv[1]
ibm1_probs_file = sys.argv[2]
# Load the IBM 1 probabilities and the lexicon.
log_info("Loading IBM 1 probs...")
ibm1_probs = load_ibm1_probs(ibm1_probs_file)

# Start training.
log_info("Starting training.")
for Dx, Dxy, chinese, english in load_parse_trees(parse_tree_dir):

    # Create the source FSA.
    src_fsa = libitg.make_fsa(chinese)

    print(Dx)

    # Featurize Dx.
    features_Dx = featurize_edges(Dx, src_fsa, ibm1_probs)
    print(features_Dx)

    # Featurize Dxy.
    # features_Dxy = featurize_edges(Dxy, src_fsa, ibm1_probs)
    break

log_info("Done training.")
