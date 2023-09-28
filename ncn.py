#!/bin/python3
# Nash Cascade  Network
# A hydrologically intuitive deep network

# Set up a solution to a network of buckets where the number of buckets in each layer
# flows out to the buckets in the next layer
# The parameter on each bucket is the size and height of each spigot.

# Need a function that solves this individually at a single buckets
# Then a function that loops through and moves the water to the downstream buckets
import torch
import json
import torch.nn as nn

G = 9.81
PRECIP_SVN = "atmosphere_water__liquid_equivalent_precipitation_rate"

class NashCascadeNetwork():

    def __init__(self, cfg_file=None, verbose=False):
        self.cfg_file = cfg_file
        self.verbose = verbose
        self.G = torch.tensor(9.81)  # Gravitational constant

    # BMI: Model Control Function
    # ___________________________________________________
    ## INITIALIZATION as required by BMI
    def initialize(self, current_time_step=0):
        # ________________________________________________________ #
        # GET VALUES FROM CONFIGURATION FILE.                      #
        self.config_from_json()
        self.initialize_up_bucket_network()
        self.precip_into_network_at_update = torch.tensor(0.0, dtype=torch.float)
        self.reset_volume_tracking()

    # ___________________________________________________
    ## CONFIGURATIONS read in from json file
    def config_from_json(self):

        with open(self.cfg_file) as cfg_file:
            cfg_loaded = json.load(cfg_file)
        self.bpl               = cfg_loaded['bpl']
        self.n_network_layers = len(self.bpl)
        self.range_of_theta_values = cfg_loaded['range_of_theta_values']
        self.initial_head_in_buckets = cfg_loaded['initial_head_in_buckets']

    # ___________________________________________________
    ## PARAMETER INITIALIZATION
    def initialize_theta_values(self):
        """ Sets random values for parameters on all spigots
        """
        # Your original network creation
        network = [[[] for _ in range(self.bpl[ilayer])] for ilayer in range(self.n_network_layers)]
        for ilayer in range(self.n_network_layers):
            n_buckets = self.bpl[ilayer]
            for ibucket in range(n_buckets):
                n_spigots = self.get_n_spigots_in_layer_buckets(ilayer)
                for ispigot in range(n_spigots):
                    random_val = self.range_of_theta_values[0] + (self.range_of_theta_values[1] - self.range_of_theta_values[0]) * torch.rand(1)
                    network[ilayer][ibucket].append(random_val)

        # Determine maximum dimensions for the tensor
        max_buckets = max(self.bpl)
        max_spigots = max([self.get_n_spigots_in_layer_buckets(ilayer) for ilayer in range(self.n_network_layers)])

        # Use a list comprehension to create a structure with the correct values and padding
        tensor_values = [[[network[ilayer][ibucket][ispigot] if (ibucket < len(network[ilayer]) and ispigot < len(network[ilayer][ibucket])) else 0 
                        for ispigot in range(max_spigots)] for ibucket in range(max_buckets)] for ilayer in range(self.n_network_layers)]

        # Convert the list to a tensor with required gradients
        self.theta = torch.tensor(tensor_values, requires_grad=True)

    # ___________________________________________________
    ## INITIAL HEAD LEVEL IN EACH BUCKET
    def initialize_bucket_head_level(self):
        for ilayer, n_buckets in enumerate(self.bpl):
            H0_tensor = torch.tensor(self.initial_head_in_buckets, dtype=torch.float32)
            self.network[ilayer]["H"] = H0_tensor.repeat(n_buckets)
        self.summarize_network()

    # ___________________________________________________
    ## BUCKET PARAMETERS
    @staticmethod
    def get_initial_parameters_of_one_bucket(n_spigots):
        """
            Args:
                n_spigots (int): The number of spigots in each bucket
            returns 
                Tensor: Tensor with shape [n_spigots, 2] (for height and area)
        """
        s_parameters = torch.rand((n_spigots, 2))
        return s_parameters
    
    # ___________________________________________________
    ## INITIAL SETUP OF THE BUCKET NETWORK
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
            n_spigots = self.get_n_spigots_in_layer_buckets(ilayer)

            # Initialize tensor with shape [n_buckets, n_spigots, 2]
            spigot_tensor = torch.zeros((n_buckets, n_spigots, 2))
            for i in range(n_buckets):
                spigot_tensor[i] = self.get_initial_parameters_of_one_bucket(n_spigots)
            
            bucket_network_dict[ilayer]["S"] = spigot_tensor

            zeros_tensor = torch.zeros((n_buckets, n_spigots))
            bucket_network_dict[ilayer]["s_q"] = zeros_tensor

            self.initialize_theta_values()

        self.network = bucket_network_dict
        self.initialize_bucket_head_level()
    
    # ___________________________________________________
    ## SOLVE THE FLOW OUT OF A SINGLE BUCKET
    def solve_single_bucket(self, H, S, theta, bucket_inflow=torch.tensor(0.0, dtype=torch.float)):
        """ Solves the flow out of a single bucket
            Args:
                H (tensor): The height of the water in a bucket before computing output
                S (list): The spigot information
                theta (tensor): The spigot coefficient
                bucket_inflow (tensor): The mass that has gone into the bucket before computing output
            Returns:
                H (tensor): The new height of the water in a bucket
                s_q (tensor): Flow out of each spigot of the bucket
        """
        H = H + bucket_inflow
        H_effective = H
        n_spigots = len(S)
        
        # Initialize s_q as a tensor with zeros
        s_q = torch.zeros(n_spigots)

        for i, (S_i, theta_i) in enumerate(zip(S, theta)):
            sp_h, sp_a = S_i

            # Here we don't want to use the full H, but we actually want to integrate from H_(t-1) to H_t
            # But we can't know H_t unless we run the calculation.
            # So unless we iterate, we can't really solve this.
            h = torch.max(torch.tensor(0.0), H_effective - sp_h)
            v = theta_i * torch.sqrt(2 * self.G * h)
            flow_out = v * sp_a
            
            # Directly assign the value to the tensor
            s_q[i] = flow_out

            # But for the lower spigots, we can simplify to calculate v as a function of half the head lost from previous spigots
            H_effective = H - torch.sum(s_q[:i+1]) / 2

        Q = torch.sum(s_q)
        # Then we need to subtract out the total lost head
        H = H - Q

        return H, s_q

    # ___________________________________________________
    ## INFLOWS TO EACH LAYER
    def solve_layer_inflows(self, ilayer):
        """
            Args: ilayer (int): The specific layer to solve
            Returns: inflows_tensor (tensor): The flows into the layer
        """
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

        return torch.tensor(inflows)

    # ___________________________________________________
    ## UPDATE FUNCTION FOR ONE SINGLE TIME STEP
    def update_network(self):
        """ Updates all the buckets with the inflow from other buckets, outflow and input from precip
        """
        for ilayer in list(self.network.keys()):

            inflows = self.solve_layer_inflows(ilayer)

            for ibucket in range(len(self.network[ilayer]["H"])):
                H = self.network[ilayer]["H"][ibucket]
                S = self.network[ilayer]["S"][ibucket]

                theta = self.theta[ilayer][ibucket]

                single_bucket_inflow = inflows[ibucket]
                H, s_q = self.solve_single_bucket(H, S, theta, single_bucket_inflow)

                self.network[ilayer]["H"][ibucket] = H  # Assuming H is a scalar or this assignment is correct

                if s_q.shape == self.network[ilayer]["s_q"][ibucket].shape:
                    self.network[ilayer]["s_q"][ibucket] = s_q
                else:
                    raise ValueError("Shapes of s_q and the target tensor are not the same.")

        network_outflow = torch.sum(self.network[list(self.network.keys())[-1]]["s_q"])
        self.network_outflow = network_outflow

        self.precipitation_mass_into_network += self.precip_into_network_at_update
        self.mass_out_of_netowork += self.network_outflow


    # ___________________________________________________
    ## SUMMARIZE THE MASS HEIGHTS ACROSS NETWORK LAYER
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

    # ___________________________________________________
    ## SET A SPECIFIC VALUE
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
            self.precip_into_network_at_update = torch.sum(src)

    # ___________________________________________________
    ## NUMBER OF SPIGOTS FOR EACH BUCKET IN NETWORK LAYER
    def get_n_spigots_in_layer_buckets(self, ilayer):
        """ Solves the number of spigots needed to flow into downstream bucket layer
            Args: ilayer (int): The specific layer of the network
            Returns: n_spigots (int): The number of spigots per bucket
        """
        if ilayer < self.n_network_layers - 1:
            n_spigots = self.bpl[ilayer+1]
        else:
            n_spigots = 1
        return n_spigots

    # ___________________________________________________
    ## SANITY CHECK ON GRADIENT VALUE
    def check_the_gradient_value_on_theta(self, function_str):
        print(f"WARNING: Checking Gradients {function_str}: self.theta.grad", self.theta.grad)
        # if not self.theta.grad:
        #     print(f"WARNING: Checking Gradients {function_str}: self.theta.grad", self.theta.grad)

    # ________________________________________________
    # Mass balance tracking
    def reset_volume_tracking(self):
        self.summarize_network()
        self.inital_mass_in_network = torch.sum(torch.tensor([tensor.item() for tensor in self.sum_H_per_layer]))
        self.precipitation_mass_into_network = 0
        self.mass_out_of_netowork = 0

    # ________________________________________________
    # Mass balance tracking
    def report_out_mass_balance(self):
        self.summarize_network()
        mass_in_network = torch.sum(torch.stack(self.sum_H_per_layer))
        print(f"Mass in network: {mass_in_network:.1f}")
        print(f"Total Mass out of network {self.mass_out_of_netowork:.1f}")
        print(f"Total precipitation into network {self.precipitation_mass_into_network:.1f}")
        inital_mass_plus_precipitation = (self.inital_mass_in_network + self.precipitation_mass_into_network)
        mass_left_in_network_plus_mass_out = (mass_in_network + self.mass_out_of_netowork)
        mass_balance = inital_mass_plus_precipitation - mass_left_in_network_plus_mass_out
        print(f"Mass balance for network is {mass_balance:.3f}")
        inital_mass_less_mass_left = (self.inital_mass_in_network - mass_in_network) 
        mass_out_less_precip_in = (self.mass_out_of_netowork - self.precipitation_mass_into_network)
        mass_balance = inital_mass_less_mass_left - mass_out_less_precip_in
        print(f"Mass balance for network is {mass_balance:.3f}")

    # ________________________________________________
    # Forward 
    def run_ncn_sequence(self, u):
        N_TIMESTEPS = u.shape[0]
        y_pred_list = []
        for i in range(N_TIMESTEPS):
            self.set_value(PRECIP_SVN, u[i])
            self.update_network()
            y_pred_list.append(self.network_outflow)
        y_pred = torch.stack(y_pred_list)
        return y_pred
    
    # ________________________________________________
    # Detach everything    
    def detach_ncn_from_graph(self):
#        self.theta = self.theta.detach()
        self.precip_into_network_at_update = self.precip_into_network_at_update.detach()
        self.network_outflow = self.network_outflow.detach()
        self.inital_mass_in_network = self.inital_mass_in_network.detach()
        for layer in list(self.network.keys()):
            for j in list(self.network[layer].keys()):
                self.network[layer][j] = self.network[layer][j].detach()
        self.inital_mass_in_network = self.inital_mass_in_network.detach()
        self.G = self.G.detach()

# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
if __name__ == "__main__":
    PRECIP_SVN = "atmosphere_water__liquid_equivalent_precipitation_rate"
    DO_PLOT = False
    N_TIMESTEPS = 127
    network_precip_input_list = []
    count = 0
    for i in range(N_TIMESTEPS):
        ###########################################################################
        if count > 39:
            network_precip_input_list.append(1.0)
        else:
            network_precip_input_list.append(0.0)
        if count == 50:
            count = 0
        count+=1
    network_precip_tensor = torch.tensor(network_precip_input_list)
    total_mass_precip_in = torch.sum(network_precip_tensor)
    ###########################################################################

    ###########################################################################
    ###########################################################################
    # Example 0
    bucket_net0 = NashCascadeNetwork(cfg_file="./config_0.json")
    bucket_net0.initialize()
    bucket_net0.summarize_network()
    inital_mass_in_network0 = torch.sum(torch.tensor([tensor.item() for tensor in bucket_net0.sum_H_per_layer]))

    network_outflow_tensor_0 = bucket_net0.run_ncn_sequence(network_precip_tensor)

    final_mass_in_network0 = torch.sum(torch.stack(bucket_net0.sum_H_per_layer))
    bucket_net0.report_out_mass_balance()
