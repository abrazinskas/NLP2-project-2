# this file contains an example how to run the model
import os
from models.CRF import CRF
from misc.helper import load_ibm1_probs
from misc.utils import extend_forest_with_rules_by_rhs, create_batches
from misc.featurizer import Featurizer
from misc.embeddings import WordEmbeddings

# parameters
learning_rate = 1e-6
epochs = 3
batch_size = 50

# paths
parse_tree_dir = "data/parses_small/"  # format:[ source, target, Dx, Dxy, Dnx]
ibm1_probs_file_path = "data/lexicon"
embeddings_dir = "data/embeddings/"
word_embeddings_ch_file_path = os.path.join(embeddings_dir, "zh.pkl")
word_embeddings_en_file_path = os.path.join(embeddings_dir, "en.pkl")
word_clusters_ch_file_path = os.path.join(embeddings_dir, "clusters.zh")
word_clusters_en_file_path = os.path.join(embeddings_dir, "clusters.en")

# loading extra params for the model
ibm1_probs = load_ibm1_probs(ibm1_probs_file_path)
embeddings_ch = WordEmbeddings(word_embeddings_ch_file_path, word_clusters_ch_file_path)
embeddings_en = WordEmbeddings(word_embeddings_en_file_path, word_clusters_en_file_path)

featurizer = Featurizer(ibm1_probs, embeddings_ch, embeddings_en)


crf = CRF(learning_rate=learning_rate)

for epoch in range(1, epochs+1):
    print("epoch %d" % epoch)
    for j, batch in enumerate(create_batches(parse_tree_dir, batch_size=batch_size)):

        for Dx, Dxy, chinese, english in batch:
            # add extra attribute that use in outside computations
            # TODO: move it to some better place
            extend_forest_with_rules_by_rhs(Dx)
            extend_forest_with_rules_by_rhs(Dxy)

        # load featurizer ( TODO: make a simple function call in the class itself )
        crf.features = featurizer.featurize_parse_trees_batch(batch)

        print('using data from line %d' % (j+1))
        # ll_before = crf.compute_loglikelihood(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
        ll_before = crf.compute_loglikelihood_batch(batch=batch)
        print("log-likelihood before %f" % ll_before)

        # crf.train(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
        crf.train_batch(batch=batch)

        # ll_after = crf.compute_loglikelihood(source_sentence=chinese, Dxy=Dxy, Dnx=Dx)
        ll_after = crf.compute_loglikelihood_batch(batch=batch)
        print("log-likelihood after %f" % ll_after)
        print("-------------------------------")

        # rules, terminals = crf.decode(source_sentence=chinese, Dnx=Dx)

