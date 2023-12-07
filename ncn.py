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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

G = 9.81
PRECIP_SVN = "atmosphere_water__liquid_equivalent_precipitation_rate"
PRECIP_SVN_SEQ = "atmosphere_water__liquid_equivalent_precipitation_rate_seq"
PRECIP_RECORD = "atmosphere_water__liquid_equivalent_precipitation_rate_record"

class NashCascadeNetwork(nn.Module):

    def __init__(self, cfg_file=None, verbose=False):
        super(NashCascadeNetwork, self).__init__()
        self.cfg_file = cfg_file
        self.verbose = verbose
        self.G = 9.8
        self.unit_precip = 6.0

    # BMI: Model Control Function
    # ___________________________________________________
    ## INITIALIZATION as required by BMI
    def initialize(self):
        # ________________________________________________________ #
        # GET VALUES FROM CONFIGURATION FILE.                      #
        self.config_from_json()
        self.initialize_up_bucket_network()
        self.precip_into_network_at_update = torch.tensor(0.0, dtype=torch.float, requires_grad=False)
        self.reset_volume_tracking()
        if self.do_predict_theta_with_lstm:
            self.lstm_hidden_size = 16  # Set according to your needs
            self.lstm_num_layers = 1    # Number of LSTM layers
            self.initialize_LSTM()

    # ___________________________________________________
    ## INITIALIZE THE LSTM
    def initialize_LSTM(self):
        H_tensor = self.get_the_H_tensor(normalize=True)
        self.input_u_sequence_length = self.n_network_layers * 2

        # LSTM input size is the size of H_tensor plus one for u
        lstm_input_size = H_tensor.numel() + 1  # Adding one for the u feature
        self.linear = nn.Linear(self.lstm_hidden_size, self.theta.numel())
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers
        )

        # Initialize hidden state and cell state
        self.hidden = (torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size),
                    torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size))

    # ___________________________________________________
    ## CONFIGURATIONS read in from json file
    def config_from_json(self):

        with open(self.cfg_file) as cfg_file:
            cfg_loaded = json.load(cfg_file)
        self.bpl               = cfg_loaded['bpl']
        self.n_network_layers = len(self.bpl)
        self.range_of_theta_values = cfg_loaded['range_of_theta_values']
        self.initial_head_in_buckets = cfg_loaded['initial_head_in_buckets']
        self.precip_distribution = cfg_loaded['precip_distribution']
        torch.cuda.manual_seed(cfg_loaded['seed'])
        torch.manual_seed(cfg_loaded['seed'])
        if cfg_loaded['verbose'] == "True":
            self.verbose = True
        else:
            self.verbose = False
        if cfg_loaded['do_initialize_random_theta_values'] == "True":
            self.do_initialize_random_theta_values = True
        else:
            self.do_initialize_random_theta_values = False
        if cfg_loaded['do_predict_theta_with_lstm'] == "True":
            self.do_predict_theta_with_lstm = True
        else:
            self.do_predict_theta_with_lstm = False

        self.model_type = cfg_loaded['model_type']
        #-----------------------------------------------------------#
        if self.model_type == "NCNN":
            self.train_learning_rate = cfg_loaded['train_learning_rate']
            self.train_lr_step_size = cfg_loaded['train_lr_step_size']
            self.train_lr_gamma = cfg_loaded['train_lr_gamma']
            self.epochs = cfg_loaded['epochs']

    # ___________________________________________________
    ## PARAMETER INITIALIZATION
    def initialize_theta_values(self):
        """ Sets random values for parameters on all spigots
        """
        values_list = []
        for ilayer in range(self.n_network_layers):
            n_buckets = self.bpl[ilayer]
            for ibucket in range(n_buckets):
                n_spigots = self.get_n_spigots_in_layer_buckets(ilayer)
                for ispigot in range(n_spigots):
                    random_val = self.range_of_theta_values[0] + (self.range_of_theta_values[1] - self.range_of_theta_values[0]) * torch.rand(1)
                    values_list.append(random_val.item())  # Convert tensor to scalar and append

        # Convert the list to a 1D tensor
        self.theta = torch.tensor(values_list, requires_grad=True)

    # ___________________________________________________
    ## INITIAL HEAD LEVEL IN EACH BUCKET
    def initialize_bucket_head_level(self):
        for ilayer, n_buckets in enumerate(self.bpl):
#            H0_tensor = torch.tensor(self.initial_head_in_buckets, dtype=torch.float32, requires_grad=True)
            H0_tensor = torch.tensor(self.initial_head_in_buckets, dtype=torch.float32, requires_grad=False)
            self.network[ilayer]["H"] = H0_tensor.repeat(n_buckets)
        self.summarize_network()
    
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
#            spigot_tensor = torch.rand((n_buckets, n_spigots, 2), requires_grad=True)
            spigot_tensor = torch.rand((n_buckets, n_spigots, 2), requires_grad=False)

            bucket_network_dict[ilayer]["S"] = spigot_tensor

#            zeros_tensor = torch.zeros((n_buckets, n_spigots), requires_grad=True)
            zeros_tensor = torch.zeros((n_buckets, n_spigots), requires_grad=False)
            bucket_network_dict[ilayer]["s_q"] = zeros_tensor

        if self.do_initialize_random_theta_values:
            self.initialize_theta_values()
        self.network = bucket_network_dict
        self.initialize_bucket_head_level()
    
    # ___________________________________________________
    ## SOLVE THE FLOW OUT OF A SINGLE BUCKET
    def solve_single_bucket(self, H, S, theta_indices, bucket_inflow):
        """ Solves the flow out of a single bucket
            Args:
                H (tensor): The height of the water in a bucket before computing output
                S (tensor): The spigot information
                theta_indices (tensor): The spigot coefficient indexes
                bucket_inflow (tensor): The mass that has gone into the bucket before computing output
            Returns:
                H (tensor): The new height of the water in a bucket
                s_q (tensor): Flow out of each spigot of the bucket
        """
        H_initial_local = H
        H_effective = H_initial_local + bucket_inflow
        n_spigots = S.shape[0]
        
        # Initialize s_q as a tensor with zeros
        s_q_local = torch.zeros(n_spigots)

        for i, (S_i, theta_i) in enumerate(zip(S, self.theta[theta_indices])):

            sp_h, sp_a = S_i

            # Here we don't want to use the full H, but we actually want to integrate from H_(t-1) to H_t
            # But we can't know H_t unless we run the calculation.
            # So unless we iterate, we can't really solve this.
            h = torch.max(torch.tensor(0.0), H_effective - sp_h)
            v = theta_i * torch.sqrt(2 * self.G * h)
            flow_out = v * sp_a
            
            # Directly assign the value to the tensor
            s_q_local[i] = flow_out

            # But for the lower spigots, we can simplify to calculate v as a function of half the head lost from previous spigots
            H_effective = H_initial_local - torch.sum(s_q_local[:i+1]) / 2

        # Ensure that the denominator is not zero to avoid division by zero
        denominator = torch.sum(s_q_local)
        denominator = torch.where(denominator == 0, torch.tensor(1.0), denominator)

        # Calculate the multiplier
        multiplier = torch.min(torch.tensor(1.0), (H_initial_local + bucket_inflow) / denominator)

        # Ensure the multiplier is on the same device as the other tensors (e.g., GPU if they are on GPU)
        multiplier = multiplier.to(H_initial_local.device)

        s_q_local = s_q_local * multiplier

        H_final = H_initial_local - torch.sum(s_q_local) + bucket_inflow
        H_final = torch.clamp(H_final, min=0)

        if H_final < -0.001:
            print("---------------WARNING---------------")
            print("H_final is leq zero", H_final)
            print("s_q_local is", s_q_local)
        
        return H_final, s_q_local

    # ___________________________________________________
    ## INFLOWS TO EACH LAYER
    def solve_layer_inflows(self, ilayer):
        """
            Args: ilayer (int): The specific layer to solve
            Returns: inflows_tensor (tensor): The flows into the layer
            Notes: if self.precip_distribution == "upstream", then all the precip goes only into the top bucket
                   elif self.precip_distribution == "even", then it is evenly distributed amungst the layers
        """
        precip_inflow_per_layer = self.precip_into_network_at_update / self.n_network_layers

        if ilayer == 0:
            if self.precip_distribution == "upstream":
                inflows = [self.precip_into_network_at_update]
            elif self.precip_distribution == "even":
                inflows = [precip_inflow_per_layer]
            return inflows
        
        number_of_ilayer_buckets = self.network[ilayer]["s_q"].shape[0]
        number_of_upstream_buckets = self.network[ilayer-1]["s_q"].shape[0]
        inflows = []

        for ibucket in range(number_of_ilayer_buckets):

            if self.precip_distribution == "upstream":
                precip_inflow_per_bucket = 0
            elif self.precip_distribution == "even":
                precip_inflow_per_bucket = precip_inflow_per_layer/number_of_ilayer_buckets

            i_bucket_q = 0
            for i_up_bucket in range(number_of_upstream_buckets):
                i_bucket_q += self.network[ilayer-1]["s_q"][i_up_bucket][ibucket]

            inflows.append(i_bucket_q + precip_inflow_per_bucket)

        return torch.tensor(inflows, requires_grad=True)
        
    # ___________________________________________________
    ## Get the theta index, which is the number of upstream layers, buckets and spigots
    def get_theta_indices_for_bucket(self, ilayer, ibucket):
        """Compute the indices in self.theta for the given ilayer and ibucket."""
        
        # For layers before the current layer
        preceding_layers_sum = sum(self.bpl[i] * self.get_n_spigots_in_layer_buckets(i) for i in range(ilayer))
        
        # For the starting spigot of the current bucket in the current layer
        start_index = preceding_layers_sum + ibucket * self.get_n_spigots_in_layer_buckets(ilayer)
        
        # Number of spigots in the current bucket
        n_spigots = self.get_n_spigots_in_layer_buckets(ilayer)
        
        # Create a list of indices for the thetas corresponding to the spigots of the current bucket
        indices = list(range(start_index, start_index + n_spigots))
        
        return indices

    # ___________________________________________________
    ## PREPARE LSTM INPUTS
    def prepare_lstm_inputs(self):
        H_tensor = self.get_the_H_tensor(normalize=True)
        # Normalize the precipitation sequence
        u_sequence = self.precip_seq_into_network_at_update / self.unit_precip
        # Ensure u_sequence is a 1D tensor [sequence_length]
        u_sequence_1d = u_sequence.view(-1)
        # Reshape H_tensor to [1, number_of_features]
        H_tensor_2d = H_tensor.view(1, -1)
        # Repeat H_tensor for each time step in u_sequence
        H_tensor_repeated = H_tensor_2d.repeat(u_sequence_1d.size(0), 1)
        # Reshape u_sequence to [sequence_length, 1]
        u_sequence_2d = u_sequence_1d.unsqueeze(1)
        # Concatenate u_sequence with H_tensor_repeated
        lstm_input = torch.cat((u_sequence_2d, H_tensor_repeated), dim=1)
        # Ensure lstm_input is [sequence_length, 1, input_size]
        lstm_input = lstm_input.view(u_sequence_1d.size(0), 1, -1)
        return lstm_input

    # ___________________________________________________
    ## UPDATE FUNCTION FOR ONE SINGLE TIME STEP
    def update(self):
        """ Updates all the buckets with the inflow from other buckets, outflow and input from precip
        """

        if self.do_predict_theta_with_lstm:
            lstm_input = self.prepare_lstm_inputs()
            # Pass through LSTM
            lstm_output, _ = self.lstm(lstm_input)
            lstm_output = self.linear(lstm_output)  # Get the last time step output for the linear layer
            lstm_output = self.sigmoid(lstm_output)  # Apply the sigmoid activation
            # Post-process LSTM output to match the shape of self.theta
            # Assuming lstm_output is [1, sequence_length, hidden_size] and self.theta is [hidden_size]
            lstm_output = lstm_output.squeeze(0)  # remove batch dimension
            self.theta = lstm_output[-1][0]  # take the last timestep

            lstm_output.retain_grad()
            self.lstm_output = lstm_output
        
        for ilayer in list(self.network.keys()):

            inflows = self.solve_layer_inflows(ilayer)

            s_q_local = self.network[ilayer]["s_q"]

            number_of_ilayer_buckets = self.network[ilayer]["H"].shape[0]
            for ibucket in range(number_of_ilayer_buckets):
                H_local_in = self.network[ilayer]["H"][ibucket]
                S_local_in = self.network[ilayer]["S"][ibucket]

                theta_indices = self.get_theta_indices_for_bucket(ilayer, ibucket)

                single_bucket_inflow = inflows[ibucket]
                H_local_out, s_q_out = self.solve_single_bucket(H_local_in, S_local_in, theta_indices, single_bucket_inflow)

                self.network[ilayer]["H"][ibucket] = H_local_out 

                s_q_local[ibucket] = s_q_out

            self.network[ilayer]["s_q"] = s_q_local

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
        if name == "atmosphere_water__liquid_equivalent_precipitation_rate_seq":
            self.precip_seq_into_network_at_update = src
        if name == "atmosphere_water__liquid_equivalent_precipitation_rate_record":
            self.precip_record = src

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
        print(f"Initial Mass in network: {self.inital_mass_in_network:.1f}")
        print(f"Final Mass in network: {mass_in_network:.1f}")
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
    # Get the bucket heights into a single tensor
    def get_the_H_tensor(self, normalize=False):
        H_list = []
        for i in list(self.network.keys()):
            H_list.extend(list(self.network[i]['H']))
        H_tensor = torch.tensor(H_list)
        if normalize:
            H_tensor = H_tensor / 10.0
        return H_tensor

    # ________________________________________________
    # SET SEQUENCE PRECIP
    def set_sequence_precip(self, u, i):
        if self.do_predict_theta_with_lstm:
            if i >= self.input_u_sequence_length:
                # If 'i' is large enough, simply take the required sequence from 'u'
                sequence = u[i-self.input_u_sequence_length:i]
            else:
                # If 'i' is too small, pad 'u' up to 'i' with zeros at the beginning
                padding_size = self.input_u_sequence_length - i
                padding = u.new_zeros((padding_size,))
                sequence = torch.cat((padding, u[:i]), dim=0)
            self.set_value(PRECIP_SVN_SEQ, sequence)

    # ________________________________________________
    # Forward 
    def forward(self):

        u = self.precip_record
        END_u = u.shape[0]
        y_pred_list = []
        for i in range(END_u):
            self.set_value(PRECIP_SVN, u[i])

            self.set_sequence_precip(u, i)

            self.update()

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



# ___________________________________________________
## PARAMETER TUNING
def train_model(model, u, y_true):
    """ This function is used to update the theta values to minimize the loss function
        Args:
            model (ncnn): NashCascadeNeuralNetwork
            u (tensor): precipitation input
            y_true (tensor): true predictions

    """
    model.set_value(PRECIP_RECORD, u.clone().detach())

    model.do_initialize_random_theta_values = False
    
    # Instantiate the loss function
    criterion = nn.MSELoss()
    # We will collect all parameters from both the LSTM and Linear layer for optimization
    lstm_params = model.lstm.parameters()
    linear_params = model.linear.parameters()
    # Combine all parameters into a single list
    all_params = list(lstm_params) + list(linear_params)

    # Create the optimizer instance with the combined parameter list
    optimizer = optim.SGD(all_params, lr=model.train_learning_rate)

    scheduler = StepLR(optimizer, step_size=model.train_lr_step_size, gamma=model.train_lr_gamma, verbose=True)

    for epoch in range(model.epochs):

#        model.initialize_up_bucket_network()

        optimizer.zero_grad()

        # FORWARD PASS OF THE MODEL
        y_pred = model.forward()

        start_criterion = int(y_pred.shape[0]/2)
        loss = criterion(y_pred[start_criterion:], y_true[start_criterion:])

        loss.backward() # run backpropagation

        optimizer.step() # update the parameters, just like w -= learning_rate * w.grad

        print(f"loss: {loss:.4f}------------")
        print(f"theta: {model.theta}")
        print("model.lstm_output.grad[-1][0]")
        print(model.lstm_output.grad[-1][0])

        model.detach_ncn_from_graph()

        model.first_lstm_prediction = True

        scheduler.step()

    return y_pred, loss



# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#
if __name__ == "__main__":
    N_TIMESTEPS = 1
    network_precip_input_list = []
    count = 0
    unit_precip = 6.0
    for i in range(N_TIMESTEPS):
        ###########################################################################
        if i == 0:
            network_precip_input_list.append(unit_precip)
        else:
            network_precip_input_list.append(0.0)
    network_precip_tensor = torch.tensor(network_precip_input_list)
    total_mass_precip_in = torch.sum(network_precip_tensor)
        ###########################################################################

    ###########################################################################
    ###########################################################################
    # Example 0
    bucket_net0 = NashCascadeNetwork(cfg_file="./config_0.json")
    bucket_net0.initialize()
    bucket_net0.set_value(PRECIP_RECORD, torch.tensor(network_precip_input_list, requires_grad=False))
    bucket_net0.summarize_network()
    inital_mass_in_network0 = torch.sum(torch.tensor([tensor.item() for tensor in bucket_net0.sum_H_per_layer]))
    network_outflow_tensor_0 = bucket_net0.forward()
    final_mass_in_network0 = torch.sum(torch.stack(bucket_net0.sum_H_per_layer))
    bucket_net0.report_out_mass_balance()

    ###########################################################################
    ###########################################################################
    # Train theta values of Network1
    cfg_file="./config_1.json"
    bucket_net1 = NashCascadeNetwork(cfg_file=cfg_file)
    bucket_net1.initialize()
    bucket_net1.set_value(PRECIP_RECORD, torch.tensor(network_precip_input_list, requires_grad=False))
    bucket_net1.unit_precip = unit_precip
    y_pred, loss = train_model(bucket_net1, network_precip_tensor.detach(), network_outflow_tensor_0.detach())
    bucket_net1.report_out_mass_balance()
    if N_TIMESTEPS <= 5:
        from torchviz import make_dot
        make_dot(loss).view()