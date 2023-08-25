# Nash Cascade Neural Network
# A hydrologically intuitive deep learning network

# Set up a solution to a network of buckets where the number of buckets in each layer
# flows out to the buckets in the next layer
# The parameter on each bucket is the size and height of each spigot.

# Need a function that solves this individually at a single buckets
# Then a function that loops through and moves the water to the downstream buckets
import numpy as np
import torch
import json
G = 9.81

class NashCascadeNeuralNetwork:

    def __init__(self, cfg_file=None, verbose=False):
        self.cfg_file = cfg_file
        self.verbose = verbose
        self.G = 9.81  # Gravitational constant

    # BMI: Model Control Function
    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def initialize(self,current_time_step=0):
        # ________________________________________________________ #
        # GET VALUES FROM CONFIGURATION FILE.                      #
        self.config_from_json()
        self.initialize_up_bucket_network()

    #________________________________________________________
    def config_from_json(self):

        with open(self.cfg_file) as cfg_file:
            cfg_loaded = json.load(cfg_file)
        # ___________________________________________________
        ## MANDATORY CONFIGURATIONS
        self.bpl               = cfg_loaded['bpl']
        self.range_of_alpha_values = cfg_loaded['range_of_alpha_values']
        self.initial_head_in_buckets = cfg_loaded['initial_head_in_buckets']


    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    @staticmethod
    def get_initial_parameters_of_one_bucket(n_spigots):
        """
            Args:
                n_spigots (int): The number of spigots in each bucket
            returns 
                list: [[height, area],[height, area],...,[height, area]]
        """
        s_parameters = [[np.random.random(), np.random.random()] for _ in range(n_spigots)]
        return s_parameters

    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def initialize_up_bucket_network(self):
        """Sets up the network of buckets
        Args: 
            bpl (list): the buckets per layer
        Returns:
            dict: A dictionary with all the buckets organized by layer, 
                  with each layer containing a dictionary, 
                  with keys for Head in the bucket and a list of spigot parms
        """

        bucket_network_dict = {layer: {"H": [], "S": [], "s_q": []} for layer in range(len(self.bpl))}

        for ilayer, n_buckets in enumerate(self.bpl):

            if ilayer < len(self.bpl)-1:
                n_spigots = self.bpl[ilayer+1]
            else:
                n_spigots = 1

            spigots = [self.get_initial_parameters_of_one_bucket(n_spigots) for _ in range(n_buckets)]

            bucket_network_dict[ilayer]["S"] = spigots

            H0 = self.initial_head_in_buckets

            bucket_network_dict[ilayer]["H"] = [H0 for _ in range(n_buckets)]

            bucket_network_dict[ilayer]["s_q"] = [[0 for i in range(n_spigots)] for _ in range(n_buckets)]

            alpha1, alpha2 = self.range_of_alpha_values

            bucket_network_dict[ilayer]["theta"] = [[np.random.uniform(alpha1, alpha2) for i in range(n_spigots)] for _ in range(n_buckets)]

        self.network = bucket_network_dict
    
    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def solve_single_bucket(self, H, S, theta, bucket_inflow=0):
        H = H + bucket_inflow
        s_q = []
        for S_i, theta_i in zip(S, theta):
            sp_h, sp_a = S_i
            h = np.max([0, H - sp_h])
            v = theta_i * np.sqrt(2 * self.G * h)
            flow_out = v * sp_a
            s_q.append(flow_out)
        Q = np.sum(s_q)
        H = H - Q
        return H, s_q

    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def solve_layer_inflows(self, ilayer):
        if ilayer == 0:
            inflows = [0]
            return inflows
        number_of_ilayer_buckets = len(self.network[ilayer]["s_q"])
        number_of_upstream_buckets = len(self.network[ilayer-1]["s_q"])
        inflows = []
        for ibucket in range(number_of_ilayer_buckets):
            i_bucket_q = 0
            for i_up_bucket in range(number_of_upstream_buckets):
                i_bucket_q += self.network[ilayer-1]["s_q"][i_up_bucket][ibucket]
            inflows.append(i_bucket_q)
        return inflows

    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def update_network(self):
        for ilayer in list(self.network.keys()):
            inflows = self.solve_layer_inflows(ilayer)
            for ibucket in range(len(self.network[ilayer]["H"])):
                H = self.network[ilayer]["H"][ibucket]
                S = self.network[ilayer]["S"][ibucket]
                theta = self.network[ilayer]["theta"][ibucket]
                single_bucket_inflow = inflows[ibucket]
                H, s_q = self.solve_single_bucket(H, S, theta, single_bucket_inflow)
                self.network[ilayer]["H"][ibucket] = H
                self.network[ilayer]["s_q"][ibucket] = s_q
        network_outflow = self.network[list(self.network.keys())[-1]]["s_q"][0]
        self.network_outflow = network_outflow

    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def summarize_network(self):
        mean_H_per_layer = []
        sum_H_per_layer = []
        for i in list(self.network.keys()):
            mean_H_per_layer.append(np.mean(self.network[i]["H"]))
            sum_H_per_layer.append(np.sum(self.network[i]["H"]))
        self.mean_H_per_layer = mean_H_per_layer
        self.sum_H_per_layer = sum_H_per_layer
