import sys
import numpy as np
from collections import defaultdict

if len(sys.argv) < 3:
    print("Use: python parse_lexicon.py <lexicon> <output> [top_n]")
    sys.exit()

lexicon_file = sys.argv[1]
output_file = sys.argv[2]
top_n = 5 if len(sys.argv) < 4 else int(sys.argv[3])

translation_candidates = defaultdict(list)
with open(lexicon_file) as f:
    for line in f:
        chinese, english, p_e_given_f, p_f_given_e = line.split()

        if chinese == "<NULL>":
            chinese = "-EPS-"

        if english == "<NULL>":
            english = "-EPS-"

        p_e_given_f = 0.0 if p_e_given_f == "NA" else float(p_e_given_f)
        p_f_given_e = 0.0 if p_f_given_e == "NA" else float(p_f_given_e)
        translation_score = np.sqrt(p_e_given_f * p_f_given_e + 1e-10)
        translation_candidates[chinese].append((english, translation_score))

with open(output_file, "w+") as f:
    for word, scores in translation_candidates.items():
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_n_words = [x[0] for x in sorted_scores[:top_n]]
        f.write("%s -> %s\n" % (word, ' '.join(top_n_words)))
