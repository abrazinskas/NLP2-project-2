# this file contains an example how to run the model
import pickle
from models.CRF import CRF
from misc.helper import load_lexicon, scan_line, featurize_edges, load_ibm1_probs, load_parse_trees
from misc.utils import extend_forest_with_rules_by_rhs

# data_file_path = "data/grammars/dev2.processed"  # format:[ source, target, Dx, Dxy, Dnx]
parse_tree_dir = "data/parses/"  # format:[ source, target, Dx, Dxy, Dnx]
ibm1_probs_file_path = "data/lexicon"

ibm1_probs = load_ibm1_probs(ibm1_probs_file_path)
crf = CRF(ibm1_probs=ibm1_probs)

for Dx, Dxy, chinese, english in load_parse_trees(parse_tree_dir):
    # ll = crf.compute_loglikelihood(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
    # print("log-likelihood %f" % ll)

    # add extra attribute
    extend_forest_with_rules_by_rhs(Dx)
    extend_forest_with_rules_by_rhs(Dxy)
    crf.train(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
