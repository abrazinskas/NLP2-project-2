import numpy as np
import os, sys
import lib.libitg as libitg
from lib.libitg import Symbol, Terminal, Nonterminal, Span
from lib.libitg import Rule, CFG
from lib.libitg import FSA
from misc.helper import load_ibm1_probs, log_info, load_parse_trees, load_dev_data, load_lexicon
from misc.featurizer import Featurizer
from misc.embeddings import WordEmbeddings
from misc.utils import create_batches, extend_forest_with_rules_by_rhs
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from models.CRF import CRF

# Parameters.
learning_rate = 1e-6
num_epochs = 1
batch_size = 50

# Arguments.
if len(sys.argv) < 5:
    print("Use: python parse_training_data.py <parse_tree_dir> <ibm1_probs> <embeddings_dir> <output_dir>")
    sys.exit()

# Data files.
parse_tree_dir = sys.argv[1]
ibm1_probs_file = sys.argv[2]

embeddings_dir = sys.argv[3]
word_embeddings_file_ch = os.path.join(embeddings_dir, "zh.pkl")
word_embeddings_file_en = os.path.join(embeddings_dir, "en.pkl")
word_clusters_file_ch = os.path.join(embeddings_dir, "clusters.zh")
word_clusters_file_en = os.path.join(embeddings_dir, "clusters.en")

output_dir = sys.argv[4]
viterbi_output_file = os.path.join(output_dir, "viterbi.out")
mbr_output_file = os.path.join(output_dir, "mbr.out")

# Load the IBM 1 probabilities and the lexicon.
log_info("Loading IBM 1 probs...")
ibm1_probs = load_ibm1_probs(ibm1_probs_file)

# Load the word embeddings.
log_info("Loading word embeddings...")
embeddings_ch = WordEmbeddings(word_embeddings_file_ch, word_clusters_file_ch)
embeddings_en = WordEmbeddings(word_embeddings_file_en, word_clusters_file_en)

# Create the featurizer and the CRF.
featurizer = Featurizer(ibm1_probs, embeddings_ch, embeddings_en)
crf = CRF(learning_rate=learning_rate)

# Start training.
log_info("Starting training.")
for epoch in range(1, num_epochs + 1):
    log_info("Epoch %d" % epoch)
    log_info("========")

    for batch_num, batch in enumerate(create_batches(parse_tree_dir, batch_size=batch_size)):
        log_info("Batch %d" % batch_num)

        for Dx, Dxy, chinese, english in batch:
            extend_forest_with_rules_by_rhs(Dx)
            extend_forest_with_rules_by_rhs(Dxy)

        # Compute features for this training instance.
        features = featurizer.featurize_parse_trees_batch(batch)
        crf.features = features

        # Compute the log-likelihood before the batch.
        ll_before = crf.compute_loglikelihood_batch(batch=batch)
        log_info("Log-likelihood before = %f" % ll_before)

        # Train on the batch.
        crf.train_batch(batch=batch)

        # Report the new log-likelihood on this batch.
        ll_after = crf.compute_loglikelihood_batch(batch=batch)
        log_info("Log-likelihood after = %f" % ll_after)
        log_info("--------")
        break
log_info("Done training.")

# TODO what we choose here as top_n may affect our results. In the small
# training set we use top_n=5, so we use it here too.
lexicon = load_lexicon("data/sorted_translations", top_n=5)
val_data = "data/data/dev1.zh-en"
test_data = "data/data/dev2.zh-en"

# Decode on the validation set and report validation BLEU.
all_refs = []
hypotheses = [] 
for chinese, references, Dx in load_dev_data(val_data, lexicon):
    crf.features = featurizer.featurize_parse_trees(Dx, None, chinese)
    viterbi_y = crf.decode_viterbi(source_sentence=chinese, Dnx=Dx)
    cur_refs = [r.split() for r in references]
    hypotheses.append(viterbi_y)
    all_refs.append(cur_refs)
val_bleu = corpus_bleu(all_refs, hypotheses, \
        smoothing_function=SmoothingFunction().method7)
log_info("Validation BLEU: %f" % val_bleu)

# Calculate the model score in terms of BLEU on the test set and write
# the decoded strings to a file.
with open(os.path.join(output_dir, "test_pred.txt"), "w+") as f:
    all_refs = []
    hypotheses = [] 
    for chinese, references, Dx in load_dev_data(test_data, lexicon):
        crf.features = featurizer.featurize_parse_trees(Dx, None, chinese)
        viterbi_y = crf.decode_viterbi(source_sentence=chinese, Dnx=Dx)
        cur_refs = [r.split() for r in references]
        hypotheses.append(viterbi_y)
        all_refs.append(cur_refs)
        f.write("%s\n" % " ".join(viterbi_y))

test_bleu = corpus_bleu(all_refs, hypotheses, \
        smoothing_function=SmoothingFunction().method7)
log_info("Test BLEU: %f" % test_bleu)

# Do inference on the test set and write the results both using
# Viterbi and MBR decoding.
# log_info("Performing inference on the test set...")
# for temp in [1.0, 1.2, 1.6, 2.0]:
#     crf.temp = temp
#     with open(viterbi_output_file + "-%f" % temp, "w+") as vof, open(mbr_output_file + "-%f" % temp, "w+") as mof:
#         for j, batch in enumerate(create_batches(parse_tree_dir, batch_size=batch_size)):
#             crf.features = featurizer.featurize_parse_trees_batch(batch)
#             for Dx, Dxy, chinese, english in batch:
#                 rules, terminals = crf.decode(source_sentence=chinese, Dnx=Dx)
#                 vof.write("\t".join([english, chinese, " ".join(terminals)])+"\n")
#                 mbr_deriv = crf._decode_mbr(chinese, Dx, 100)
#                 mof.write("\t".join([english, chinese, " ".join(mbr_deriv)])+"\n")
#             break
#     log_info("Viterbi decoding results written to %s" % viterbi_output_file)
#     log_info("MBR decoding results written to %s" % mbr_output_file)
