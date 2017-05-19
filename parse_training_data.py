import dill as pickle
import sys, os
import lib.libitg as libitg
from lib.libitg import Symbol, Terminal, Nonterminal, Span
from lib.libitg import Rule, CFG
from lib.libitg import FSA
from misc.helper import load_lexicon, scan_line, log_info

if len(sys.argv) < 4:
    print("Use: python parse_training_data.py <training_data> <lexicon> <output_dir>")
    sys.exit()

# Data files.
training_data = sys.argv[1]
lexicon = sys.argv[2]
output_dir = sys.argv[3]
dx_dir = "Dx"
dxy_dir = "Dxy"

# Hyperparameters.
max_length = 10
top_n = 5
log_info("Hyperparams: max_length=%s, top_n=%s" % (max_length, top_n))

log_info("Loading lexicon...")
lexicon = load_lexicon(lexicon, top_n=top_n)

log_info("Creating output folders if necessary...")
dxy_path = os.path.join(output_dir, dxy_dir)
dx_path = os.path.join(output_dir, dx_dir)
english_file = os.path.join(output_dir, "english")
chinese_file = os.path.join(output_dir, "chinese")

if not os.path.exists(dx_path):
    log_info("Creating %s" % dx_path)
    os.makedirs(dx_path)

if not os.path.exists(dxy_path):
    log_info("Creating %s" % dxy_path)
    os.makedirs(dxy_path)

total = 0
covered = 0
with open(training_data) as f, open(english_file, "w+") as en_file, open(chinese_file, "w+") as ch_file:
    for line in f:
        total += 1
        chinese, english = scan_line(line)

        # Skip training sentences longer than some max_length if max_length is set.
        if max_length is not None and len(chinese) > max_length:
            continue

        sub_lexicon = {k:lexicon[k] for k in chinese.split() + ["-EPS-"] if k in lexicon}
        src_cfg = libitg.make_source_side_finite_itg(sub_lexicon)

        # Create an FSA for the source sentence and parse the source sentence.
        src_fsa = libitg.make_fsa(chinese)
        _Dx = libitg.earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), \
                sprime_symbol=Nonterminal("D(x)"), clean=True)
        Dx = libitg.make_target_side_itg(_Dx, lexicon)

        # Create a target FSA and D_i(x, y)
        tgt_fsa = libitg.make_fsa(english)
        Dxy = libitg.earley(Dx, tgt_fsa, start_symbol=Nonterminal("D(x)"), \
                sprime_symbol=Nonterminal('D(x,y)'), clean=True)

        # Continue in the case the parse for this training sentence is empty.
        if len(Dx) == 0 or len(Dxy) == 0:
            continue

        # Store Dix.
        with open(os.path.join(dx_path, str(covered)), "wb") as f:
            pickle.dump(Dx, f)

        # # Store Dixy.
        with open(os.path.join(dxy_path, str(covered)), "wb") as f:
            pickle.dump(Dxy, f)

        # Store the english and chinese sentence.
        en_file.write("%s\n" % english)
        ch_file.write("%s\n" % chinese)

        covered += 1
        log_info("%d/%d trees extracted." % (covered, total))

print("Training instances covered: %d/%d" % (covered, total))
