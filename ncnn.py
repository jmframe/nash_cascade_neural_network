# Nash Cascade Neural Network
# A hydrologically intuitive deep learning network

# Set up a solution to a network of buckets where the number of buckets in each layer
# flows out to the buckets in the next layer
# The parameter on each bucket is the size and height of each spigot.

# Need a function that solves this individually at a single buckets
# Then a function that loops through and moves the water to the downstream buckets
import torch
import json

G = 9.81

class NashCascadeNeuralNetwork:

    def __init__(self, cfg_file=None, verbose=False):
        self.cfg_file = cfg_file
        self.verbose = verbose
        self.G = torch.tensor(9.81)  # Gravitational constant

    # BMI: Model Control Function
    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def initialize(self,current_time_step=0):
        # ________________________________________________________ #
        # GET VALUES FROM CONFIGURATION FILE.                      #
        self.config_from_json()
        self.initialize_up_bucket_network()
        self.precip_into_network_at_update = 0

    #________________________________________________________
    def config_from_json(self):

        with open(self.cfg_file) as cfg_file:
            cfg_loaded = json.load(cfg_file)
        # ___________________________________________________
        ## MANDATORY CONFIGURATIONS
        self.bpl               = cfg_loaded['bpl']
        self.n_network_layers = len(self.bpl)
        self.range_of_theta_values = cfg_loaded['range_of_theta_values']
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
        s_parameters = [[torch.rand(1), torch.rand(1)] for _ in range(n_spigots)]
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

        bucket_network_dict = {layer: {"H": None, "S": None, "s_q": None} for layer in range(len(self.bpl))}

        for ilayer, n_buckets in enumerate(self.bpl):

            if ilayer < self.n_network_layers - 1:
                n_spigots = self.bpl[ilayer+1]
            else:
                n_spigots = 1

            spigots = [self.get_initial_parameters_of_one_bucket(n_spigots) for _ in range(n_buckets)]
            bucket_network_dict[ilayer]["S"] = spigots

            H0_tensor = torch.tensor(self.initial_head_in_buckets, dtype=torch.float32, requires_grad=True)
            bucket_network_dict[ilayer]["H"] = H0_tensor.repeat(n_buckets)

            zeros_tensor = torch.zeros(n_buckets, n_spigots, requires_grad=True)
            bucket_network_dict[ilayer]["s_q"] = zeros_tensor

            theta1, theta2 = self.range_of_theta_values
            if ilayer in [0, self.n_network_layers-1]:
                bucket_network_dict[ilayer]["theta"] = torch.full((n_buckets, n_spigots), theta2, requires_grad=True)
            else:
                bucket_network_dict[ilayer]["theta"] = torch.rand(n_buckets, n_spigots) * (theta2 - theta1) + theta1
                bucket_network_dict[ilayer]["theta"].requires_grad_()

        self.network = bucket_network_dict
    
    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def solve_single_bucket(self, H, S, theta, bucket_inflow=0):
        H = H + bucket_inflow
        s_q = []
        for S_i, theta_i in zip(S, theta):
            sp_h, sp_a = S_i
            h = torch.max(torch.tensor(0.0), H - sp_h)
            v = theta_i * torch.sqrt(2 * self.G * h)
            flow_out = v * sp_a
            s_q.append(flow_out)
        Q = torch.sum(torch.stack(s_q))
        H = H - Q
        return H, s_q

    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def solve_layer_inflows(self, ilayer):

        precip_inflow_per_layer = self.precip_into_network_at_update / self.n_network_layers

        if ilayer == 0:
            inflows = [precip_inflow_per_layer]
            return inflows
        
        number_of_ilayer_buckets = len(self.network[ilayer]["s_q"])
        number_of_upstream_buckets = len(self.network[ilayer-1]["s_q"])
        inflows = []

        for ibucket in range(number_of_ilayer_buckets):

            precip_inflow_per_bucket = precip_inflow_per_layer/number_of_ilayer_buckets

            i_bucket_q = 0
            for i_up_bucket in range(number_of_upstream_buckets):
                i_bucket_q += self.network[ilayer-1]["s_q"][i_up_bucket][ibucket]
            inflows.append(i_bucket_q + precip_inflow_per_bucket)

        return inflows

    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def update_network(self):
        """ Updates all the buckets with the inflow from other buckets, outflow and input from precip
        """
        for ilayer in list(self.network.keys()):

            inflows = self.solve_layer_inflows(ilayer)

            for ibucket in range(len(self.network[ilayer]["H"])):
                H = self.network[ilayer]["H"][ibucket]
                S = self.network[ilayer]["S"][ibucket]
                theta = self.network[ilayer]["theta"][ibucket]
                single_bucket_inflow = inflows[ibucket]
                H, s_q = self.solve_single_bucket(H, S, theta, single_bucket_inflow)
                self.network[ilayer]["H"][ibucket] = H  # Assuming H is a scalar or this assignment is correct

                # Now we have to set s_q, which has been difficult with the tensor situation
                s_q_tensor = torch.tensor(s_q, dtype=self.network[ilayer]["s_q"][ibucket].dtype)
                if s_q_tensor.shape == self.network[ilayer]["s_q"][ibucket].shape:
                    # Convert the network tensor to a list
                    current_list = self.network[ilayer]["s_q"].tolist()
                    # Update the list with new values
                    current_list[ibucket] = s_q_tensor.tolist()
                    # Convert the list back to a tensor
                    self.network[ilayer]["s_q"] = torch.tensor(current_list, dtype=s_q_tensor.dtype, requires_grad=True)
                else:
                    raise ValueError("Shapes of s_q and the target tensor are not the same.")

        network_outflow = torch.sum(self.network[list(self.network.keys())[-1]]["s_q"])
        self.network_outflow = network_outflow


    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def summarize_network(self):
        """ Gets some summary of each layer of the network, mean and sum
        """
        mean_H_per_layer = []
        sum_H_per_layer = []

        for i in list(self.network.keys()):
            mean_H_per_layer.append(torch.mean(self.network[i]["H"]))
            sum_H_per_layer.append(torch.sum(self.network[i]["H"]))

        self.mean_H_per_layer = mean_H_per_layer
        self.sum_H_per_layer = sum_H_per_layer

    # -----------------------------------------------------------------------#
    #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN #### NCNN ####
    # -----------------------------------------------------------------------#
    def set_value(self, name,  src):
        """Specify a new value for a model variable.

        This is the setter for the model, used to change the model's
        current state. It accepts, through *src*, a new value for a
        model variable, with the type, size and rank of *src*
        dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        src : array_like
            The new value for the specified variable.
        """
        if name == "atmosphere_water__liquid_equivalent_precipitation_rate":
            
            self.precip_into_network_at_update = torch.sum(torch.tensor(src))