# this file contains an example how to run the model
import os
from models.CRF import CRF
from misc.helper import load_ibm1_probs, load_lexicon
from misc.utils import extend_forest_with_rules_by_rhs, create_batches, get_run_var
from misc.featurizer import Featurizer
from misc.embeddings import WordEmbeddings
from misc.log import Log
import time

# parameters
learning_rate = 1e-6
regul_strength = 1e-3
epochs = 1
batch_size = 50
decay_rate = 1000.

# paths
parse_tree_dir = "data/parses_small/"  # format:[ source, target, Dx, Dxy, Dnx]
ibm1_probs_file_path = "data/lexicon"
embeddings_dir = "data/embeddings/"
word_embeddings_ch_file_path = os.path.join(embeddings_dir, "zh.pkl")
word_embeddings_en_file_path = os.path.join(embeddings_dir, "en.pkl")
word_clusters_ch_file_path = os.path.join(embeddings_dir, "clusters.zh")
word_clusters_en_file_path = os.path.join(embeddings_dir, "clusters.en")
output_folder_path = "output/"
output_folder_path = os.path.join(output_folder_path, str(get_run_var(output_folder_path)))


# loading extra params for the model
ibm1_probs = load_ibm1_probs(ibm1_probs_file_path)
embeddings_ch = WordEmbeddings(word_embeddings_ch_file_path, word_clusters_ch_file_path)
embeddings_en = WordEmbeddings(word_embeddings_en_file_path, word_clusters_en_file_path)

log = Log(output_folder_path)

# write experimental setup
log.write("EXPERIMENTAL SETUP: ")
log.write("parse_tree_dir %s" % parse_tree_dir)
log.write("lexicon_file_path %s" % ibm1_probs_file_path)
log.write("learning rate %f" % learning_rate)
log.write("regul. strength %f" % regul_strength)
log.write("batch size %d" % batch_size)
log.write("decay rate %f" % decay_rate)
log.write("-------------------------------")


featurizer = Featurizer(ibm1_probs, embeddings_ch, embeddings_en)

crf = CRF(learning_rate=learning_rate, regul_strength=regul_strength, decay_rate=decay_rate)

for epoch in range(1, epochs+1):
    start = time.time()
    log.write("epoch %d" % epoch)
    for j, batch in enumerate(create_batches(parse_tree_dir, batch_size=batch_size)):

        for Dx, Dxy, chinese, english in batch:
            # add extra attribute that use in outside computations
            # TODO: move it to some better place
            extend_forest_with_rules_by_rhs(Dx)
            extend_forest_with_rules_by_rhs(Dxy)

        # load featurizer ( TODO: make a simple function call in the class itself )
        crf.features = featurizer.featurize_parse_trees_batch(batch)

        log.write('using data from batch # %d' % (j+1))
        # ll_before = crf.compute_loglikelihood(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
        ll_before = crf.compute_loglikelihood_batch(batch=batch)
        log.write("log-likelihood before %f" % ll_before)

        # crf.train(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
        crf.train_batch(batch=batch)

        ll_after = crf.compute_loglikelihood_batch(batch=batch)
        log.write("log-likelihood after %f" % ll_after)
        log.write("-------------------------------")

    end = time.time()
    log.write("epoch completed in %f minutes " % ((end - start)/60.0))


# perform inference and save in a file
translations_file = open(os.path.join(output_folder_path, "translations.txt"), 'w')
print('performing inference')
for j, batch in enumerate(create_batches(parse_tree_dir, batch_size=batch_size)):
    # load featurizer TODO: ( make a simple function call in the class itself )
    crf.features = featurizer.featurize_parse_trees_batch(batch)
    for Dx, Dxy, chinese, english in batch:
        terminals = crf.decode_viterbi(source_sentence=chinese, Dnx=Dx)
        translations_file.write("\t".join([english, chinese, " ".join(terminals)])+"\n")
translations_file.close()
print('done')





