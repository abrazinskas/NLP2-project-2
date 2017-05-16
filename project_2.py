import libitg
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from libitg import FSA
from collections import defaultdict

def load_lexicon(lexicon_file):
    lexicon = defaultdict(set)
    with open(lexicon_file) as f:
        for line in f:
            line_split = line.split(" -> ")
            assert len(line_split) == 2
            source = line_split[0]
            targets = line_split[1].split()
            lexicon[source].update(targets + ["-EPS-"])
    return lexicon

lexicon = load_lexicon("f_to_e.txt")
src_cfg = libitg.make_source_side_itg(lexicon)
src_fsa = libitg.make_fsa('爸爸 宏伟')
src_forest = libitg.earley(src_cfg, src_fsa, \
               start_symbol=Nonterminal('S'), \
               sprime_symbol=Nonterminal("D(x)"))
Dx = libitg.make_target_side_itg(src_forest, lexicon)
tgt_fsa = libitg.make_fsa('magnificent dad')
Dxy = libitg.earley(Dx, tgt_fsa,
                    start_symbol=Nonterminal("D(x)"), 
                    sprime_symbol=Nonterminal('D(x,y)'))
