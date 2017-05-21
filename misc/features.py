from lib.libitg import Span, Nonterminal, Rule

class Features():

    def __init__(self):
        self.edge2fmap = dict()

    def _extract_rule_repr(self, rule):
        lhs = rule.lhs
        rhs = list(rule.rhs)

        # Remove bispans from the lhs.
        if isinstance(rule.lhs, Span):
            s, _, _ = rule.lhs.obj()
            if isinstance(s, Span) or s == Nonterminal("D(x)"):
                lhs = s

        # And the same for the rhs.
        for i in range(len(rule.rhs)):
            if isinstance(rule.rhs[i], Span):
                s, _, _ = rule.rhs[i].obj()
                if isinstance(s, Span):
                        rhs[i] = s

        new_rule = Rule(lhs, tuple(rhs))
        return new_rule

    def add(self, edge, fmap):
        key = self._extract_rule_repr(edge)
        self.edge2fmap[key] = fmap

    def get(self, edge):
        key = self._extract_rule_repr(edge)
        return self.edge2fmap[key]

