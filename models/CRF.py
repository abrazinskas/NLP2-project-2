import pickle
import os
from pickle import UnpicklingError
import lib.libitg as libitg
from misc.features_old import featurize_edge
from models.support import inside_algorithm, outside_algorithm, top_sort, expected_feature_vector
import numpy as np

np.random.seed(1)


class CRF():

    def __init__(self, ibm1_probs, learning_rate=1e-8):
        self.parameters = {}  # initialization is performed as a feature function is observed
        self.ibm1_probs = ibm1_probs
        self.learning_rate = learning_rate

    def train(self, source_sentence, Dxy, Dnx):
        src_fsa = libitg.make_fsa(source_sentence)

        # 1. compute expectations
        Dnx_inside, _ = self.compute_inside_values(Dnx, src_fsa)
        Dxy_inside, _ = self.compute_inside_values(Dxy, src_fsa)
        Dnx_outside = self.compute_outside_values(Dnx, src_fsa, Dnx_inside)
        Dxy_outside = self.compute_outside_values(Dxy, src_fsa, Dxy_inside)

        edge_features = lambda rule: self.get_feature(rule, src_fsa)

        first_expectation = expected_feature_vector(forest=Dxy, inside=Dxy_inside, outside=Dxy_outside,
                                                     edge_features=edge_features)
        # seems to be a problem here!
        # TODO: fix it! 
        second_expectation = expected_feature_vector(forest=Dnx, inside=Dnx_inside, outside=Dnx_outside,
                                                    edge_features=edge_features)

        # print("-------------")
        # print("length of first expected feature vector is %d" % len(first_expectation))
        # print("length of second expected feature vector is %d" % len(second_expectation))

        # 2. update parameters
        for feature_name in second_expectation.keys():
            derivative = - second_expectation[feature_name]
            if feature_name in first_expectation:
                derivative += first_expectation[feature_name]
            current_weight_value = self.get_parameter(feature_name)
            self.parameters[feature_name] = current_weight_value + self.learning_rate * derivative


    def compute_loglikelihood(self, source_sentence, Dxy, Dnx):
        src_fsa = libitg.make_fsa(source_sentence)
        _, log_Zxy = self.compute_inside_values(Dxy, src_fsa)
        _, log_Znx = self.compute_inside_values(Dnx, src_fsa)
        return log_Zxy - log_Znx

    def weight_function(self, edge, src_fsa) -> float:
        features = self.get_feature(edge, src_fsa)
        dot_product = 0
        for feature_name in features.keys():
            dot_product += features[feature_name] * self.get_parameter(feature_name)
        return dot_product

    def compute_inside_values(self, grammar, src_fsa):
        """
        Wrapper function that computes inside values, returns both inside values and log normalizer
        """
        sorted_nodes = top_sort(grammar)
        root_node = sorted_nodes[-1]
        edge_weights = {rule: self.weight_function(rule, src_fsa) for rule in grammar._rules}
        inside_values = inside_algorithm(grammar, sorted_nodes, edge_weights=edge_weights)
        return inside_values, inside_values[root_node]

    def compute_outside_values(self, grammar, src_fsa, inside_values):
        """
        Wrapper function that computes outside values
        """
        # TODO: make a separate function that computes both inside and outside to avoid sorting and computing
        # TODO: edge values multiple times.
        sorted_nodes = top_sort(grammar)
        edge_weights = {rule: self.weight_function(rule, src_fsa) for rule in grammar._rules}
        return outside_algorithm(grammar, sorted_nodes, edge_weights, inside_values)

    def get_feature(self, edge, src_fsa):
        return featurize_edge(edge=edge, src_fsa=src_fsa, ibm1_probs=self.ibm1_probs)

    def get_parameter(self, feature_name):
        """
        Returns a parameter that corresponds to the feature_name. If parameter has not be initialized previously, it will
        initialize it.
        """
        if feature_name not in self.parameters:
            self.parameters[feature_name] = np.random.normal()
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
