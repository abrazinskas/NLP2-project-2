import pickle
import sys
import numpy as np

from misc.helper import log_info
from sklearn.cluster import MiniBatchKMeans

if len(sys.argv) < 4:
    print("Use: python create_clusters.py <embeddings> <nclusters> <output_file>")
    sys.exit()

filename = sys.argv[1]
nclusters = int(sys.argv[2])
output_file = sys.argv[3]

log_info("Loading embeddings")
with open(filename, "rb") as f:
    unpickler = pickle._Unpickler(f)
    unpickler.encoding = "latin1"
    words, embeddings = unpickler.load()
X = np.array([embedding for embedding in embeddings])

log_info("Starting clustering...")
kmeans = MiniBatchKMeans(n_clusters=nclusters, n_init=10, batch_size=350, init_size=1050).fit(X)

log_info("Saving cluster indices in %s" % output_file)
with open(output_file, "w+") as f:
    for i, cluster_id in enumerate(kmeans.labels_):
        f.write("%s\n" % (cluster_id))
