import numpy as np
import os, sys
import lib.libitg as libitg
from lib.libitg import Symbol, Terminal, Nonterminal, Span
from lib.libitg import Rule, CFG
from lib.libitg import FSA
from misc.helper import load_ibm1_probs, log_info, load_parse_trees
from misc.features import featurize_edges, featurize_edge
from misc.embeddings import WordEmbeddings

if len(sys.argv) < 4:
    print("Use: python parse_training_data.py <parse_tree_dir> <ibm1_probs> <embeddings_dir>")
    sys.exit()

# Data files.
parse_tree_dir = sys.argv[1]
ibm1_probs_file = sys.argv[2]
embeddings_dir = sys.argv[3]
word_embeddings_file_ch = os.path.join(embeddings_dir, "zh.pkl")
word_embeddings_file_en = os.path.join(embeddings_dir, "en.pkl")
word_clusters_file_ch = os.path.join(embeddings_dir, "clusters.zh")
word_clusters_file_en = os.path.join(embeddings_dir, "clusters.en")

# Load the IBM 1 probabilities and the lexicon.
log_info("Loading IBM 1 probs...")
ibm1_probs = load_ibm1_probs(ibm1_probs_file)

# Load the word embeddings.
log_info("Loading word embeddings...")
embeddings_ch = WordEmbeddings(word_embeddings_file_ch, word_clusters_file_ch)
embeddings_en = WordEmbeddings(word_embeddings_file_en, word_clusters_file_en)

# Start training.
log_info("Starting training.")
for Dx, Dxy, chinese, english in load_parse_trees(parse_tree_dir):

    # Create the source FSA.
    src_fsa = libitg.make_fsa(chinese)

    # Featurize Dx, and add one rule that only occurs in Dxy.
    features = featurize_edges(Dx, src_fsa, ibm1_probs)
    features.add(Dxy._rules[-1], featurize_edge(Dxy._rules[-1], src_fsa, ibm1_probs))

log_info("Done training.")
