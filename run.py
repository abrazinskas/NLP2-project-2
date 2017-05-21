# this file contains an example how to run the model
import pickle
from models.CRF import CRF
from misc.helper import load_ibm1_probs, load_parse_trees
from misc.utils import extend_forest_with_rules_by_rhs

# data_file_path = "data/grammars/dev2.processed"  # format:[ source, target, Dx, Dxy, Dnx]
parse_tree_dir = "data/parses_small/"  # format:[ source, target, Dx, Dxy, Dnx]
ibm1_probs_file_path = "data/lexicon"

ibm1_probs = load_ibm1_probs(ibm1_probs_file_path)
crf = CRF(ibm1_probs=ibm1_probs, learning_rate=1e-5)

for j, (Dx, Dxy, chinese, english) in enumerate(load_parse_trees(parse_tree_dir)):
    # add extra attribute
    extend_forest_with_rules_by_rhs(Dx)
    extend_forest_with_rules_by_rhs(Dxy)
    print('using data from line %d' % (j+1))
    ll_before = crf.compute_loglikelihood(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
    print("log-likelihood before %f" % ll_before)

    crf.train(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)

    ll_after = crf.compute_loglikelihood(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
    print("log-likelihood after %f" % ll_after)
    print("-------------------------------")

