import nltk
import numpy as np
from scipy.misc import logsumexp
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# A loss function that simply calculates (1 - BLEU(r, c)).
def bleu_loss(reference, candidate):
    return 1.0 - sentence_bleu([reference], candidate, \
            smoothing_function=SmoothingFunction().method7)

# Performs Minimum Bayes Risk decoding using a given loss function.
# Expects the root of a translation hypergraph together with
# the inside values for the nodes in the tree.
def MBR_decoding(forest, root, I, num_samples, loss_fn=bleu_loss):

    # Do ancestral sampling to get some sample derivations.
    samples = ancestral_sampling(forest, root, I, num_samples)

    # Calculate the yields of the sampled derivations.
    candidates = [target_yield(sample) for sample in samples]

    # Compute the loss function for each candidate.
    candidate_loss = np.zeros(num_samples)
    for i, candidate in enumerate(candidates):
        l = [loss_fn(candidate, c) for c in candidates]
        candidate_loss[i] = np.sum(l)

    # Return the one that has minimum loss.
    return candidates[np.argmin(candidate_loss)]

# Performs ancestral sampling on a forest and returns num_samples
# derivation samples.
def ancestral_sampling(forest, root, I, num_samples):
    samples = []
    losses = []
    for i in range(num_samples):
        node_queue = [root]
        sample = []
        while len(node_queue) > 0:
            cur_node = node_queue[0]

            # There's nothing to expand anymore for terminal nodes.
            if cur_node.is_terminal():
                del node_queue[0]
                continue

            # Calculate the probabilities for all possible rhs rules
            # using the inside values.
            FS = forest._rules_by_lhs[cur_node]
            probs = np.array([log_weight_rule(rule, I) for rule in FS])
            probs -= logsumexp(probs)
            probs = np.exp(probs)

            # Sample a rule to use.
            sampled_id = np.random.choice(np.arange(len(probs)), p=probs)
            sampled_rule = FS[sampled_id]
            sample.append(sampled_rule)

            # Add the RHS of the rule to the nodes to expand and remove
            # the current node from that list.
            del node_queue[0]
            for node in sampled_rule.rhs:
                node_queue.append(node)

        samples.append(sample)
    return samples

# Returns the cumulative inside weights of a rule.
def log_weight_rule(rule, I):
    p_prime = np.array([I[node] for node in rule.rhs])
    return logsumexp(p_prime)

# Returns the target yield of a derivation.
def target_yield(derivation):
    sentence_length = derivation[0].rhs[0].obj()[2]
    result = [None for _ in range(sentence_length)]
    for rule in derivation:
        if rule.rhs[0].is_terminal():
            symbol, idx, _ = rule.rhs[0].obj()
            word = symbol.obj()
            result[idx] = word
    return result
