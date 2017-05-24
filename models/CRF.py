import pickle
import os
from pickle import UnpicklingError
import lib.libitg as libitg
from lib.formal import Span
from misc.mbr import MBR_decoding
from misc.inside_outside import inside_algorithm, outside_algorithm
from misc.support import top_sort, expected_feature_vector, viterbi_decoding, traverse_back_pointers,\
    compute_learning_rate
import numpy as np
from misc.utils import sort_hash_by_key

np.random.seed(1)


class CRF():

    def __init__(self, learning_rate=1e-8, regul_strength=1e-8, decay_rate=1e-2):
        self.parameters = {}  # initialization is performed as a feature function is observed
        self.features = None
        self.learning_rate = learning_rate
        self.regul_strength = regul_strength
        self.decay_rate = decay_rate
        self.current_step = 0  # track how many times we've updated our parameters

    def compute_gradient(self, source_sentence, Dxy, Dnx):
        src_fsa = libitg.make_fsa(source_sentence)

        # 1. compute expectations
        Dnx_inside, _ = self.compute_inside_values(Dnx, src_fsa)
        Dxy_inside, _ = self.compute_inside_values(Dxy, src_fsa)
        Dnx_outside = self.compute_outside_values(Dnx, src_fsa, Dnx_inside)
        Dxy_outside = self.compute_outside_values(Dxy, src_fsa, Dxy_inside)

        edge_features = lambda rule: self.get_features(rule, src_fsa)

        first_expectation = expected_feature_vector(forest=Dxy, inside=Dxy_inside, outside=Dxy_outside,
                                                     edge_features=edge_features)
        second_expectation = expected_feature_vector(forest=Dnx, inside=Dnx_inside, outside=Dnx_outside,
                                                    edge_features=edge_features)
        derivatives = {}
        # 2. update parameters
        for feature_name in second_expectation.keys():
            derivative = - second_expectation[feature_name]
            if feature_name in first_expectation:
                derivative += first_expectation[feature_name]
            derivatives[feature_name] = derivative
        return derivatives

    def train_batch(self, batch):
        """
        Training on a batch of instances with SGD, notice that we divide gradients by the number of data-points.
        In addition, we perform regularization.
        :param batch: (Dnx, Dxy, source, target) list
        """
        self.current_step += 1
        learning_rate = compute_learning_rate(self.learning_rate, step=self.current_step, decay_rate=self.decay_rate)
        # print(learning_rate)
        gradients_acc = {}
        # accumulate gradients
        for Dnx, Dxy, source_sentence, target_sentence in batch:
            gradient = self.compute_gradient(source_sentence, Dxy, Dnx)
            for feature_name, derivative in gradient.items():
                if feature_name not in gradients_acc:
                    gradients_acc[feature_name] = 0.
                gradients_acc[feature_name] += derivative

        # update
        for feature_name, derivative in gradients_acc.items():
            current_weight_value = self.get_parameter(feature_name)
            self.parameters[feature_name] = \
                current_weight_value + learning_rate * (derivative/len(batch) - self.regul_strength * current_weight_value)

    def compute_loglikelihood_batch(self, batch):
        """
        Computes log-likelihood on a batch of data, notice that we divide by the number of data-points
        :param batch: (Dnx, Dxy, source, target) list
        """
        loglikelihood = 0.
        for Dnx, Dxy, source_sentence, target_sentence in batch:
            loglikelihood += self.compute_loglikelihood(source_sentence, Dxy, Dnx)
        return loglikelihood/len(batch)

    def compute_loglikelihood(self, source_sentence, Dxy, Dnx):
        src_fsa = libitg.make_fsa(source_sentence)
        a, log_Zxy = self.compute_inside_values(Dxy, src_fsa)
        _, log_Znx = self.compute_inside_values(Dnx, src_fsa)
        return log_Zxy - log_Znx

    def weight_function(self, edge, src_fsa) -> float:
        features = self.get_features(edge, src_fsa)
        dot_product = 0.
        for feature_name in features.keys():
            dot_product += features[feature_name] * self.get_parameter(feature_name)
        return dot_product

    def compute_inside_values(self, grammar, src_fsa):
        """
        Wrapper function that computes inside values, returns both inside values and log normalizer
        """
        sorted_nodes, edge_weights = self.__sort_nodes_and_compute_weights(grammar, src_fsa)
        root_node = sorted_nodes[-1]
        inside_values = inside_algorithm(grammar, sorted_nodes, edge_weights=edge_weights)
        return inside_values, inside_values[root_node]

    def compute_outside_values(self, grammar, src_fsa, inside_values):
        """
        Wrapper function that computes outside values
        """
        # TODO: make a separate function that computes both inside and outside to avoid sorting and computing
        # TODO: edge values multiple times.
        sorted_nodes, edge_weights = self.__sort_nodes_and_compute_weights(grammar, src_fsa)
        return outside_algorithm(grammar, sorted_nodes, edge_weights, inside_values)

    def __sort_nodes_and_compute_weights(self, grammar, src_fsa):
        """
        This logic is preliminary to inside and outside computations (also Viterbi decoding)
        """
        sorted_nodes = top_sort(grammar)
        edge_weights = {rule: self.weight_function(rule, src_fsa) for rule in grammar._rules}
        return sorted_nodes, edge_weights

    def _decode_mbr(self, source_sentence, Dnx, num_samples):
        src_fsa = libitg.make_fsa(source_sentence)
        sorted_nodes, edge_weights = self.__sort_nodes_and_compute_weights(Dnx, src_fsa)
        root_node = sorted_nodes[-1]
        I, _ = self.compute_inside_values(Dnx, src_fsa)
        return MBR_decoding(Dnx, root_node, I, num_samples)

    def decode(self, source_sentence, Dnx, excluded_terminals=['-EPS-']):
        """
        Translates(decodes) source sentence into target sentence using Viterbi dynamic programming algorithm
        :return: :rtype: list of terminals in order
        """
        src_fsa = libitg.make_fsa(source_sentence)
        sorted_nodes, edge_weights = self.__sort_nodes_and_compute_weights(Dnx, src_fsa)
        root_node = sorted_nodes[-1]
        _, back_pointers = viterbi_decoding(Dnx, sorted_nodes, edge_weights)

        # create decoration function, and traverse back pointers recursively
        terminals_decoration_func = lambda x: x._symbol._symbol
        terminals = traverse_back_pointers(back_pointers, root_node, terminals_decoration_func)
        # do cleaning by excluding unwanted terminals
        terminals = [t for t in terminals if t not in excluded_terminals]

        return terminals

    def get_features(self, edge, src_fsa):
        return self.features.get(edge)

    def get_parameter(self, feature_name):
        """
        Returns a parameter that corresponds to the feature_name. If parameter has not be initialized previously, it will
        initialize it.
        """
        if feature_name not in self.parameters:
            self.parameters[feature_name] = np.random.normal(0, 1.)
        return self.parameters[feature_name]

    def save_parameters(self, output_dir, name='params.pkl'):
        """
        Saves parameters via pickle to a output_dir under specified name. In order for function to work, the class
        has to have self.params_to_save list of params name that is desired to save, e.g. ["theta", "gamma"]
        """
        print('writing parameters to %s folder' % output_dir)
        f = open(os.path.join(output_dir, name), 'wb')
        for param_name in self.params_to_save:
            pickle.dump([param_name, getattr(self, param_name)], f)
        f.close()
        print('done')

    def load_parameters(self, file_path):
        """
        Loads params from pickle saved file and assigns to attributes.
        The function will work for the saving made by save_parameters only
        """
        f = open(file_path, 'rb')
        print('loading parameters')
        while True:
            try:
                name, param = pickle.load(f)
                setattr(self, name, param)  # assuming that all parameters are shared variables
            except (EOFError, UnpicklingError):
                break
        f.close()
        print('done')
