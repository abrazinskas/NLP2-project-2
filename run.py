# this file contains an example how to run the model
import os
from models.CRF import CRF
from misc.helper import load_ibm1_probs
from misc.utils import extend_forest_with_rules_by_rhs, create_batches
from misc.featurizer import Featurizer
from misc.embeddings import WordEmbeddings
import time

# parameters
learning_rate = 1e-7
regul_strength = 1e-4
epochs = 8
batch_size = 50

# paths
parse_tree_dir = "data/parses_small/"  # format:[ source, target, Dx, Dxy, Dnx]
ibm1_probs_file_path = "data/lexicon"
embeddings_dir = "data/embeddings/"
word_embeddings_ch_file_path = os.path.join(embeddings_dir, "zh.pkl")
word_embeddings_en_file_path = os.path.join(embeddings_dir, "en.pkl")
word_clusters_ch_file_path = os.path.join(embeddings_dir, "clusters.zh")
word_clusters_en_file_path = os.path.join(embeddings_dir, "clusters.en")
translations_file_path = "output/translations.txt"  # output translations will be in format <source, target, translation>

# loading extra params for the model
ibm1_probs = load_ibm1_probs(ibm1_probs_file_path)
embeddings_ch = WordEmbeddings(word_embeddings_ch_file_path, word_clusters_ch_file_path)
embeddings_en = WordEmbeddings(word_embeddings_en_file_path, word_clusters_en_file_path)

featurizer = Featurizer(ibm1_probs, embeddings_ch, embeddings_en)

crf = CRF(learning_rate=learning_rate, regul_strength=regul_strength)

for epoch in range(1, epochs+1):
    start = time.time()
    print("epoch %d" % epoch)
    for j, batch in enumerate(create_batches(parse_tree_dir, batch_size=batch_size)):

        for Dx, Dxy, chinese, english in batch:
            # add extra attribute that use in outside computations
            # TODO: move it to some better place
            extend_forest_with_rules_by_rhs(Dx)
            extend_forest_with_rules_by_rhs(Dxy)

        # load featurizer ( TODO: make a simple function call in the class itself )
        crf.features = featurizer.featurize_parse_trees_batch(batch)

        print('using data from batch # %d' % (j+1))
        # ll_before = crf.compute_loglikelihood(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
        ll_before = crf.compute_loglikelihood_batch(batch=batch)
        print("log-likelihood before %f" % ll_before)

        # crf.train(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
        crf.train_batch(batch=batch)

        # ll_after = crf.compute_loglikelihood(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
        ll_after = crf.compute_loglikelihood_batch(batch=batch)
        print("log-likelihood after %f" % ll_after)
        print("-------------------------------")

    end = time.time()
    print("epoch completed in %f minutes " % ((end - start)/60.0))


# perform inference and save in a file
translations_file = open(translations_file_path, 'w')
print('doing inference')
for j, batch in enumerate(create_batches(parse_tree_dir, batch_size=batch_size)):
    # load featurizer ( TODO: make a simple function call in the class itself )
    crf.features = featurizer.featurize_parse_trees_batch(batch)
    for Dx, Dxy, chinese, english in batch:
        terminals = crf.decode_viterbi(source_sentence=chinese, Dnx=Dx)
        translations_file.write("\t".join([english, chinese, " ".join(terminals)])+"\n")
translations_file.close()
print('done')





