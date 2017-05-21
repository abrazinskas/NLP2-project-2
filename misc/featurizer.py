import numpy as np
from collections import defaultdict
from lib.libitg import Symbol, Terminal, Nonterminal, Span, Rule, FSA

import lib.libitg as libitg
from .spans import get_target_word, get_source_word, get_phrase
from .features import Features

class Featurizer():

    def __init__(self, ibm1_probs, embeddings_ch, embeddings_en):
        self.ibm1_probs = ibm1_probs
        self.embeddings_ch = embeddings_ch
        self.embeddings_en = embeddings_en

    def featurize_parse_trees(self, Dx, Dxy, x):
        src_fsa = libitg.make_fsa(x)
        features = Features()
        for edge in Dx:
            features.add(edge, self._featurize_edge(edge, src_fsa))
        features.add(Dxy._rules[-1], self._featurize_edge(Dxy._rules[-1], src_fsa))
        return features

    def _featurize_edge(self, edge, src_fsa):
        fmap = defaultdict(float)

        # Check if the edge represents a binary or unary rule.
        if len(edge.rhs) == 2:
            self._featurize_binary_rule(edge, src_fsa, fmap)
        else:

            # Check the type of rule that we're dealing with.
            if edge.rhs[0].is_terminal():
                self._featurize_terminal_rule(edge, src_fsa, fmap)
            elif edge.lhs.obj()[0] != Nonterminal("X"):
                self._featurize_start_rule(edge, src_fsa, fmap)
            else:
                self._featurize_upgrade_rule(edge, src_fsa, fmap)

        return fmap

    def _featurize_upgrade_rule(self, rule, src_fsa, fmap):
        lhs_symbol, lhs_start, lhs_end = rule.lhs.obj()
        rhs_symbol, rhs_start, rhs_end = rule.rhs[0].obj() # TODO spans prob not needed

        if lhs_symbol == Nonterminal("X"):
            if rhs_symbol == Nonterminal("T"):
                fmap["type:upgrade_t"] += 1.0
            elif rhs_symbol == Nonterminal("D"):
                fmap["type:upgrade_d"] += 1.0
            elif rhs_symbol == Nonterminal("T"):
                fmap["type:upgrade_t"] += 1.0

    def _featurize_binary_rule(self, rule, src_fsa, fmap):
        fmap['type:binary'] += 1.0

        # here we could have sparse features of the source string as a function of spans being concatenated
        lhs_symbol, lhs_start, lhs_end = rule.lhs.obj()
        rhs_symbol_1, rhs_start_1, rhs_end_1 = rule.rhs[0].obj()
        rhs_symbol_2, rhs_start_2, rhs_end_2 = rule.rhs[1].obj()

        if lhs_symbol == Nonterminal("D"):
            fmap["binary:recursive_deletion"] += 1.0
        elif lhs_symbol == Nonterminal("X"):

            # Check for invertion.
            if lhs_start == rhs_start_2:
                fmap["binary:inverted"] += 1.0
            else:
                fmap["binary:monotone"] += 1.0

                # Use the inside span of X rules to represent phrases, we use
                # average representations of word vectors for this.
                src_inside_phrase = get_phrase(src_fsa, lhs_start, lhs_end)
                assert len(src_inside_phrase) > 0
                inside_repr = np.zeros(self.embeddings_ch.dim())
                for word in src_inside_phrase:
                    inside_repr += self.embeddings_ch.get(word)

                for dim, val in enumerate(inside_repr):
                    fmap["inside-phrase-%d" % dim] = val

                # Add a representation for the outside phrase.
                src_outside_phrase = get_phrase(src_fsa, 0, lhs_start) + \
                        get_phrase(src_fsa, lhs_end, src_fsa.nb_states())
                if len(src_outside_phrase) > 0:
                    outside_repr = np.zeros(self.embeddings_ch.dim())
                    for word in src_outside_phrase:
                        outside_repr += self.embeddings_ch.get(word)

                    for dim, val in enumerate(outside_repr):
                        fmap["outside-phrase-%d" % dim] = val

        # TODO sparser features

    def _featurize_start_rule(self, rule, src_fsa, fmap):
        fmap["top"] += 1.0

    def _featurize_terminal_rule(self, rule, src_fsa, fmap):
        fmap["type:terminal"] += 1.0

        lhs_symbol, lhs_start, lhs_end = rule.lhs.obj()
        rhs_symbol, rhs_start, rhs_end = rule.rhs[0].obj() # TODO spans prob not needed

        if lhs_symbol == Nonterminal("D"):
            # Deletion of a source word.
            fmap["type:deletion"] += 1.0
            fmap["target-len"] -= 1.0

            # IBM 1 deletion probabilities.
            src_word = get_source_word(src_fsa, lhs_start, lhs_end)
            fmap["ibm1:del:logprob"] += np.log(self.ibm1_probs[(src_word, "-EPS-")] + 1e-10)

            # Sparse deletion feature for specific words.
            fmap["del:%s" % src_word] += 1.0

        elif lhs_symbol == Nonterminal("I"):
            # Insertion of a target word.
            fmap["type:insertion"] += 1.0
            fmap["target-len"] += 1.0
            tgt_word = get_target_word(rhs_symbol)

            # IBM 1 insertion probability.
            fmap["ibm1:ins:logprob"] += np.log(self.ibm1_probs[("-EPS-", tgt_word)] + 1e-10)

            # Sparse insertion feature for specific target words.
            fmap["ins:%s" % tgt_word] += 1.0

        elif lhs_symbol == Nonterminal("T"):
            # Translation of a source word into a target word.
            fmap["type:translation"] += 1.0
            fmap["target-len"] += 1.0
            src_word = get_source_word(src_fsa, lhs_start, lhs_end)
            tgt_word = get_target_word(rhs_symbol)

            # IBM 1 translation probabilities.
            fmap["ibm1:x2y:logprob"] += np.log(self.ibm1_probs[(src_word, tgt_word)] + 1e-10)
            fmap["ibm1:y2x:logprob"] += np.log(self.ibm1_probs[(tgt_word, src_word)] + 1e-10)
            fmap["ibm1:geometric:log"] += np.log(np.sqrt(self.ibm1_probs[(src_word, tgt_word)] * \
                self.ibm1_probs[(tgt_word, src_word)] + 1e-10) + 1e-10)

            # Sparse word translation features.
            fmap["trans:%s/%s" % (src_word, tgt_word)] += 1.0

            # Sparse word class translation features.
            src_class = self.embeddings_ch.get_cluster_id(src_word)
            tgt_class = self.embeddings_en.get_cluster_id(tgt_word)
            fmap["trans-class:%d/%d" % (src_class, tgt_class)] += 1.0
