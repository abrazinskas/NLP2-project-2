import numpy as np
import lib.libitg as libitg
from lib.libitg import Symbol, Terminal, Nonterminal, Span
from lib.libitg import CFG
import os
import errno



def create_folders_if_not_exist(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def get_run_var(dir):
    if not os.path.exists(dir):
        return 0
    subdirectories = get_immediate_subdirectories(dir)
    return len(subdirectories)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


# loads lexicon
def load_lexicon(lexicon_file_path):
    """
    Loads lexicon in the Bryan format and return a hash object
    """
    lexicon = {}
    with open(lexicon_file_path) as f:
        for line in f:
            chinese, english = line.split('->')
            chinese = chinese.strip()
            lexicon[chinese] = english.strip().split()
    return lexicon


def create_grammars(target_sentence, lexicon, src_forest, max_length=5):
    """
    Processing input sentences and returns D(x), D(x,y), and D_n(x) grammar sets with restricted eplison insertions count
    where x is source, y is target sentence
    :param target_sentence: a list of words
    :param source_sentence: a list of words
    """

    # Create an FSA for the source sentence and parse the source sentence.

    Dix = libitg.make_target_side_itg(src_forest, lexicon)

    # Create a target FSA and D_i(x, y)
    tgt_fsa = libitg.make_fsa(target_sentence)
    Dixy = libitg.earley(Dix, tgt_fsa, start_symbol=Nonterminal("D_n(x)"), \
            sprime_symbol=Nonterminal('D(x,y)'), clean=True)

     # `strict` controls whether the constraint is |yield(d)| == n (strict=True) or |yield(d)| <= n (strict=False)
    length_fsa = libitg.LengthConstraint(max_length, strict=False)
    Dinx = libitg.earley(Dix, length_fsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal("D_n(x)"))

    return Dix, Dixy, Dinx


def create_source_forest(source_sentence, src_cfg, max_insertions=4):
    src_fsa = libitg.make_fsa(source_sentence)
    src_forest = libitg.earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), \
            sprime_symbol=Nonterminal("D(x)"), clean=True)
    # Create an FSA that allows for some maximum amount of insertions of -EPS-.
    # Create D_i(x) such that it has this constraint on the number of insertions.
    eps_count_fsa = libitg.InsertionConstraint(max_insertions)
    src_forest = libitg.earley(src_forest, eps_count_fsa, start_symbol=Nonterminal('D(x)'), \
            sprime_symbol=Nonterminal('D_n(x)'), eps_symbol=None, clean=True)
    return src_forest


def extend_forest_with_rules_by_rhs(forest: CFG):
    """
    Extends the forest by adding _rules_by_rhs attribute
    :param forest: CFG object
    """
    by_rhs = {}
    for rule in forest._rules:
        for rhs_node in rule._rhs:
            if rhs_node not in by_rhs:
                by_rhs[rhs_node] = []
            by_rhs[rhs_node].append(rule)
    forest._rules_by_rhs = by_rhs

