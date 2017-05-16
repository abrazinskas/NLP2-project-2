import sys
from collections import defaultdict

if len(sys.argv) < 4:
    print("Use: python parse_lexicon.py <lexicon> <output_e_to_f> <output_f_to_e> [top_n]")
    sys.exit()

lexicon_file = sys.argv[1]
output_e_to_f = sys.argv[2]
output_f_to_e = sys.argv[3]
top_n = 5 if len(sys.argv) < 5 else int(sys.argv[4])

e_given_f_probs = defaultdict(list)
f_given_e_probs = defaultdict(list)
with open(lexicon_file) as f:
    for line in f:
        chinese, english, p_e_given_f, p_f_given_e = line.split()

        if chinese == "<NULL>":
            chinese = "-EPS-"

        if english == "<NULL>":
            english = "-EPS-"

        if p_e_given_f != "NA":
            e_given_f_probs[chinese].append((english, float(p_e_given_f)))

        if p_f_given_e != "NA":
            f_given_e_probs[english].append((chinese, float(p_f_given_e)))

def write_top_n(probs, top_n, output_file):
    with open(output_file, "w+") as f:
        for word, probs in probs.items():
            sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)
            top_n_words = [x[0] for x in sorted_probs[:top_n]]
            f.write("%s -> %s\n" % (word, ' '.join(top_n_words)))

write_top_n(e_given_f_probs, top_n, output_f_to_e)
write_top_n(f_given_e_probs, top_n, output_e_to_f)
