# this file contains an example how to run the model
import os
from models.CRF import CRF
from misc.helper import load_ibm1_probs
from misc.utils import extend_forest_with_rules_by_rhs, create_batches, get_run_var
from misc.featurizer import Featurizer
from misc.embeddings import WordEmbeddings
from misc.log import Log
from misc.support import evaluate
import time

# parameters
learning_rate = 1e-8
regul_strength = 1e-1
epochs = 5
batch_size = 50
decay_rate = 100.
load_params = False

# paths
parse_tree_dir = "data/train_top25/"  # format:[ source, target, Dx, Dxy, Dnx]
ibm1_probs_file_path = "data/lexicon"
embeddings_dir = "data/embeddings/"
word_embeddings_ch_file_path = os.path.join(embeddings_dir, "zh.pkl")
word_embeddings_en_file_path = os.path.join(embeddings_dir, "en.pkl")
word_clusters_ch_file_path = os.path.join(embeddings_dir, "clusters.zh")
word_clusters_en_file_path = os.path.join(embeddings_dir, "clusters.en")
output_folder_path = "output/"
output_folder_path = os.path.join(output_folder_path, str(get_run_var(output_folder_path)))
val_data_path = "data/val/parses_max_5_top_25.pkl"
test_data_path = "data/test/parses_top_25.pkl"
translations_file_path = os.path.join(output_folder_path, "translations.txt")
params_file_path = "output/21/params.pkl"


# loading extra params for the model
ibm1_probs = load_ibm1_probs(ibm1_probs_file_path)
embeddings_ch = WordEmbeddings(word_embeddings_ch_file_path, word_clusters_ch_file_path)
embeddings_en = WordEmbeddings(word_embeddings_en_file_path, word_clusters_en_file_path)

log = Log(output_folder_path)

# write experimental setup
log.write("EXPERIMENTAL SETUP: ")
log.write("parse_tree_dir: %s" % parse_tree_dir)
log.write("lexicon_file_path: %s" % ibm1_probs_file_path)
log.write("learning rate: %f" % learning_rate)
log.write("regul. strength: %f" % regul_strength)
log.write("batch size: %d" % batch_size)
log.write("decay rate: %f" % decay_rate)
log.write("load params? : %r" % load_params)
log.write("-------------------------------")


featurizer = Featurizer(ibm1_probs, embeddings_ch, embeddings_en)
crf = CRF(learning_rate=learning_rate, regul_strength=regul_strength, decay_rate=decay_rate)
# load params if set
if load_params:
    crf.load_parameters(params_file_path)

for epoch in range(1, epochs+1):
    start = time.time()
    log.write("epoch %d" % epoch)
    for j, batch in enumerate(create_batches(parse_tree_dir, batch_size=batch_size)):

        for Dx, Dxy, chinese, english in batch:
            # add extra attribute that use in outside computations
            # TODO: move it to some better place
            extend_forest_with_rules_by_rhs(Dx)
            extend_forest_with_rules_by_rhs(Dxy)

        # load features ( TODO: make a simple function call in the class itself )
        crf.features = featurizer.featurize_parse_trees_batch(batch)
        crf.train_batch(batch=batch)


        ll_after = crf.compute_loglikelihood_batch(batch=batch)
        log.write("batch's #%d log-likelihood is: %f" % (j+1, ll_after))

    val_bleu, val_loglikelihood = evaluate(crf, featurizer, val_data_path)
    log.write("validation BLEU is: %f" % val_bleu)
    log.write("validation log-likelihood is: %f" % (val_loglikelihood))

    end = time.time()
    log.write("epoch completed in %f minutes " % ((end - start)/60.0))


# Finally evaluate on test set
test_bleu = evaluate(crf, featurizer, test_data_path, compute_ll=False, translations_output_file_path=translations_file_path)
log.write("test BLEU is: %f" % test_bleu)

# save params into a folder
crf.save_parameters(output_folder_path)
