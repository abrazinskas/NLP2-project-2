import pickle
import re

class WordEmbeddings():

    def __init__(self, embeddings_file, cluster_file):
        words, embeddings = self._load_embeddings(embeddings_file)
        self.id2cluster = self._load_clusters(cluster_file)
        self.words = words
        self.embeddings = embeddings
        self.word2id = { w:i for (i, w) in enumerate(words) }
        self.digits_re = re.compile("[0-9]", re.UNICODE)
        self.emb_dim = embeddings.shape[1]

    # Returns a word embedding for a given word.
    def get(self, word):
        word = self._normalize(word)
        return self.embeddings[self.word2id[word]]

    # Returns the cluster index for a given word.
    def get_cluster_id(self, word):
        word = self._normalize(word)
        word_id = self.word2id[word]
        return self.id2cluster[word_id]

    def dim(self):
        return self.emb_dim

    # Loads the embeddings files with Python 3 compatibility.
    def _load_embeddings(self, filename):
        with open(filename, "rb") as f:
            unpickler = pickle._Unpickler(f)
            unpickler.encoding = "latin1"
            words, embeddings = unpickler.load()
        return words, embeddings

    def _load_clusters(self, filename):
        id2cluster = dict()
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                id2cluster[i] = int(line)
        return id2cluster

    # Based on http://nbviewer.jupyter.org/gist/aboSamoor/6046170.
    def _normalize(self, word):

        # Substitute digits with #.
        if not word in self.word2id:
            word = self.digits_re.sub("#", word)

        # Try different cases.
        if not word in self.word2id:
            word = self._case_normalizer(word)

        # Unknown words map to <UNK>.
        if not word in self.word2id:
            return "<UNK>"

        return word

    # Based on http://nbviewer.jupyter.org/gist/aboSamoor/6046170.
    # Check for multiple cases and see whether that word is a known
    # word. If multiple words are known, return the one with the lowest
    # index which means it is most frequent.
    def _case_normalizer(self, word):
      w = word
      lower = (self.word2id.get(w.lower(), 1e12), w.lower())
      upper = (self.word2id.get(w.upper(), 1e12), w.upper())
      title = (self.word2id.get(w.title(), 1e12), w.title())
      results = [lower, upper, title]
      results.sort()
      index, w = results[0]
      if index != 1e12:
        return w
      return word
