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
        self.precip_into_network_at_update = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        self.reset_volume_tracking()
        if self.do_predict_theta_with_fcnn:
            self.initialize_FCNN()

    def initialize_soft(self):
        self.initialize_up_bucket_network()
        self.precip_into_network_at_update = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        self.reset_volume_tracking()

    # ___________________________________________________
    ## INITIALIZE THE FCNN
    def initialize_FCNN(self):
        H_tensor = self.get_the_H_tensor(normalize=False)
        input_size = H_tensor.numel() + 1  # Size of H_tensor plus one for u

        hidden_size = 16  # Example hidden size, adjust as needed
        output_size = self.theta.numel()  # Output size should match the number of theta values

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid2 = nn.Sigmoid()

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
        if cfg_loaded['do_predict_theta_with_fcnn'] == "True":
            self.do_predict_theta_with_fcnn = True
        else:
            self.do_predict_theta_with_fcnn = False

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
            H0_tensor = torch.tensor(self.initial_head_in_buckets, dtype=torch.float32, requires_grad=True)
#            H0_tensor = torch.tensor(self.initial_head_in_buckets, dtype=torch.float32, requires_grad=False)
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
            spigot_tensor = torch.rand((n_buckets, n_spigots, 2), requires_grad=True)
#            spigot_tensor = torch.rand((n_buckets, n_spigots, 2), requires_grad=False)

            bucket_network_dict[ilayer]["S"] = spigot_tensor

            zeros_tensor = torch.zeros((n_buckets, n_spigots), requires_grad=True)
#            zeros_tensor = torch.zeros((n_buckets, n_spigots), requires_grad=False)
            bucket_network_dict[ilayer]["s_q"] = zeros_tensor

        if self.do_initialize_random_theta_values:
            self.initialize_theta_values()
        self.network = bucket_network_dict
        self.initialize_bucket_head_level()
        
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
    ## PREPARE FCNN INPUTS
    def prepare_fcnn_inputs(self):
        # Get the current state of the buckets
        H_tensor = self.get_the_H_tensor(normalize=False)
        # Normalize the current precipitation input
        u_current = self.precip_into_network_at_update / self.unit_precip
        u_current = u_current.view(-1)  # Ensure it is a 1D tensor
        # Concatenate the current state and precipitation input
        fcnn_input = torch.cat((H_tensor, u_current), dim=0)
        return fcnn_input

    # ___________________________________________________
    ## UPDATE THE SPIGOT OUTFLOW FOR A SINGLE TIMESTEP
    def update_spigot_outflow(self, H, S, theta):
        # Create a tensor to hold the theta values for each spigot
        theta_for_spigots = torch.zeros_like(S[:, :, 0])

        # Populate theta_for_spigots with the correct theta values
        theta_index = 0
        for ilayer in range(self.n_network_layers):
            for ibucket in range(self.bpl[ilayer]):
                n_spigots = self.get_n_spigots_in_layer_buckets(ilayer)
                for ispigot in range(n_spigots):
                    theta_for_spigots[sum(self.bpl[:ilayer]) + ibucket, ispigot] = theta[theta_index]
                    theta_index += 1

        # Calculate the head over the spigot for each bucket
        # Ensure h is non-negative and broadcastable with theta_for_spigots
        # Subtract spigot height from H and ensure the result is non-negative
        h = torch.max(torch.zeros_like(S[:, :, 0]), H.unsqueeze(1) - S[:, :, 0])

        # Adjust the flow rate calculation
        flow_rate_modifier = 0.5 * (torch.tanh(h) + 1)  # Shift and scale tanh to go from 0 to 1

        # Calculate velocity with the flow rate modifier
        v = theta_for_spigots * torch.sqrt(2 * self.G * h) * flow_rate_modifier

        # Calculate flow out
        flow_out = v * S[:, :, 1]  # Flow rate times spigot area

        return flow_out

    # ___________________________________________________
    ## UPDATE THE HEAD IN EACH BUCKET FOR A SINGLE TIMESTEP
    def update_head_in_buckets(self, H, bucket_inflows, bucket_outflows):
        # Calculate the total outflow from each bucket
        # Update the head in each bucket
        H_final = H + bucket_inflows - bucket_outflows
        return H_final

    # ___________________________________________________
    ## GET THE HEAD VALUES FOR EACH BUCKET
    def get_network_head(self):
        # Assuming self.network is a dictionary where each layer's head is stored
        heads = [self.network[layer]["H"] for layer in self.network]
        return torch.cat(heads, dim=0)  # Concatenate heads from all layers
    
    # ___________________________________________________
    ## GET THE SPIGOT INFORMATION
    def get_network_spigot_info(self):
        spigot_infos = [self.network[layer]["S"] for layer in self.network]
        max_size_1 = max(s.size(1) for s in spigot_infos)  # Max size in dimension 1
        max_size_2 = max(s.size(2) for s in spigot_infos)  # Max size in dimension 2

        # Pad each tensor to the max size
        padded_spigot_infos = [torch.nn.functional.pad(s, (0, max_size_2 - s.size(2), 0, max_size_1 - s.size(1))) for s in spigot_infos]

        return torch.cat(padded_spigot_infos, dim=0)

    # ___________________________________________________
    ## NETWORK INFLOW AND OUTFLOW VALUES FOR THIS TIMESTEP
    def update_network_fluxes(self, s_q):

        # Initialize tensors to store inflows and outflows for all buckets in the network
        total_buckets = sum(self.bpl)  # Total number of buckets in the network
        network_inflows = torch.zeros(total_buckets, dtype=torch.float32)
        network_outflows = torch.zeros(total_buckets, dtype=torch.float32)

        # Calculate inflows and outflows for each layer
        # Sum the outflows of each bucket as the inflows for the next bucket
        network_inflows[1:] = torch.sum(s_q[:-1], dim=1)

        # Sum the outflows from the second bucket onwards as the network outflows
        network_outflows = torch.sum(s_q, dim=1)

        # Calculate precipitation inflow and distribute it according to your network's logic
        precip_inflow = self.precip_into_network_at_update
        if self.precip_distribution == "upstream":
            # All precipitation goes into the top bucket of the first layer
            network_inflows[0] += precip_inflow
        elif self.precip_distribution == "even":
            # Precipitation is evenly distributed among all layers
            precip_inflow_per_layer = precip_inflow / self.n_network_layers
            layer_starts = torch.cumsum(torch.tensor([0] + self.bpl[:-1]), 0)
            for i, start in enumerate(layer_starts):
                end = start + self.bpl[i]
                network_inflows[start:end] += precip_inflow_per_layer / self.bpl[i]

        return network_inflows, network_outflows

    # ___________________________________________________
    ## SET THE HEAD VALUES FOR ALL THE BUCKETS
    def set_network_head(self, H_updated):
        # Update the head in each bucket of the network
        # Assuming H_updated is a flat tensor representing the updated head for all buckets
        start = 0
        for layer in self.network:
            end = start + self.network[layer]["H"].numel()
            self.network[layer]["H"] = H_updated[start:end].view_as(self.network[layer]["H"])
            start = end

    # ___________________________________________________
    ## SET THE SPIGOT OUTFLOW VALUES IN THE NETOWORK
    def set_network_outflow(self, s_q):

        start_row = 0
        for ilayer in range(self.n_network_layers):
            num_buckets = self.bpl[ilayer]
            num_spigots_this_layer = self.get_n_spigots_in_layer_buckets(ilayer)
            end_row = start_row + num_buckets

            # Extract the relevant rows for this layer
            layer_s_q = s_q[start_row:end_row, :num_spigots_this_layer]

            # Assign the extracted rows to the network
            self.network[ilayer]["s_q"] = layer_s_q

            # Update the start row for the next layer
            start_row = end_row

        # Calculate total outflow from the network
        last_layer_index = list(self.network.keys())[-1]
        self.network_outflow = torch.sum(self.network[last_layer_index]["s_q"])

    # ___________________________________________________
    ## UPDATE FUNCTION FOR ONE SINGLE TIME STEP
    def update(self):
        """ Updates all the buckets with the inflow from other buckets, outflow and input from precip
        """

        if self.do_predict_theta_with_fcnn:
            fc_input = self.prepare_fcnn_inputs()  # Assuming this prepares the input correctly for the FC network
            fc_input = fc_input.view(-1)  # Flatten the input
            # Pass through fully connected layers
            fc_output = self.fc1(fc_input)
            fc_output = self.sigmoid1(fc_output)  # Apply activation function
            fc_output = self.fc2(fc_output)
            fc_output = self.sigmoid2(fc_output)  # Apply activation function
            # Update theta
            self.theta = fc_output  # take the last timestep
            fc_output.retain_grad()
            self.fc_output = fc_output

        # Assuming H, S, and theta are structured for the entire network
        H = self.get_network_head()  # Method to get the head in all buckets across the network
        S = self.get_network_spigot_info()  # Method to get spigot info for the entire network
        theta = self.theta  # Assuming theta is already structured for the entire network

        # Calculate spigot outflow for the entire network
        s_q = self.update_spigot_outflow(H, S, theta)

        # Calculate bucket inflows for the entire network
        bucket_inflows, bucket_outflows = self.update_network_fluxes(s_q)  # Method to calculate inflows for the entire network

        # Update head in buckets for the entire network
        H_updated = self.update_head_in_buckets(H, bucket_inflows, bucket_outflows)

        # Update the network state
        self.set_network_head(H_updated)

        self.set_network_outflow(s_q)

        self.summarize_network()

        self.update_mass_balance()

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
        self.current_mass_in_network = torch.sum(torch.tensor([tensor.item() for tensor in self.sum_H_per_layer]))
        self.precipitation_mass_into_network = 0
        self.mass_out_of_netowork = 0
        self.inital_mass_plus_precipitation = 0
        self.mass_left_in_network_plus_mass_out = 0
        self.mass_balance = 0

    # ________________________________________________
    # Mass balance tracking
    def report_out_mass_balance(self):
        self.summarize_network()
        print(f"Initial Mass in network: {self.inital_mass_in_network:.1f}")
        print(f"Final Mass in network: {self.current_mass_in_network:.1f}")
        print(f"Total Mass out of network {self.mass_out_of_netowork:.1f}")
        print(f"Total precipitation into network {self.precipitation_mass_into_network:.1f}")
        self.inital_mass_plus_precipitation = (self.inital_mass_in_network + self.precipitation_mass_into_network)
        self.mass_left_in_network_plus_mass_out = (self.current_mass_in_network + self.mass_out_of_netowork)
        self.mass_balance = self.inital_mass_plus_precipitation - self.mass_left_in_network_plus_mass_out
        print(f"Mass balance for network is {self.mass_balance:.3f}")

    # ________________________________________________
    # Mass balance tracking
    def update_mass_balance(self):

        # Update the total mass out of the network
        self.mass_out_of_netowork += self.network_outflow

        # Update the total precipitation mass into the network
        self.precipitation_mass_into_network += self.precip_into_network_at_update

        # Calculate the final mass in the network
        self.current_mass_in_network = torch.sum(torch.tensor([tensor.item() for tensor in self.sum_H_per_layer]))

        # Calculate and print mass balance
        self.inital_mass_plus_precipitation = (self.inital_mass_in_network + self.precipitation_mass_into_network)
        self.mass_left_in_network_plus_mass_out = (self.current_mass_in_network + self.mass_out_of_netowork)
        self.mass_balance = self.inital_mass_plus_precipitation - self.mass_left_in_network_plus_mass_out

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
        if self.do_predict_theta_with_fcnn:
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

    # ________________________________________________
    # Check that the gradients are viable to update model
    def check_fcnn_gradients(self, print_all_gradients=False):
        if print_all_gradients:
            print("Here are all the FCNN gradients:")
            print(self.fc_output.grad)

        if self.fc_output.grad is not None:
            # Find non-zero gradients
            non_zero_grads = self.fc_output.grad != 0
            # Count the number of non-zero gradients
            num_non_zero_grads = torch.sum(non_zero_grads).item()
            print(f"Total number of non-zero gradients: {num_non_zero_grads}")
            # Get the indices of non-zero gradients
            indices = torch.nonzero(non_zero_grads).squeeze()
            print(f"Indices of non-zero gradients: {indices.tolist()}")
        else:
            print("No gradients computed (grad is None).")


# ___________________________________________________
## PARAMETER TUNING
def train_model(model, u, y_true):
    """ This function is used to update the theta values to minimize the loss function
        Args:
            model (ncnn): NashCascadeNeuralNetwork
            u (tensor): precipitation input
            y_true (tensor): true predictions

    """
    print(f"INITIAL MODEL theta: {model.theta}")

    model.set_value(PRECIP_RECORD, u.clone().detach())

    model.do_initialize_random_theta_values = False
    
    # Instantiate the loss function
    criterion = nn.MSELoss()
    # Collect all parameters from the fully connected layers for optimization
    fc_params = list(model.fc1.parameters()) + list(model.fc2.parameters())
    # Create the optimizer instance with the combined parameter list
    optimizer = optim.Adam(fc_params, lr=model.train_learning_rate)

    scheduler = StepLR(optimizer, step_size=model.train_lr_step_size, gamma=model.train_lr_gamma, verbose=True)

    for epoch in range(model.epochs):

#        model.initialize_up_bucket_network()

        optimizer.zero_grad()

        # FORWARD PASS OF THE MODEL
        y_pred = model.forward()

        start_criterion = int(y_pred.shape[0]/2)
        loss = criterion(y_pred[start_criterion:], y_true[start_criterion:])

        #with torch.autograd.set_detect_anomaly(True):
        loss.backward() # run backpropagation

        optimizer.step() # update the parameters, just like w -= learning_rate * w.grad

        print(f"loss: {loss:.4f}------------")
        print(f"theta: {model.theta}")
        model.check_fcnn_gradients(print_all_gradients=True)

        model.detach_ncn_from_graph()

        scheduler.step()

        model.initialize_soft()

    # FORWARD PASS OF THE MODEL
    y_pred = model.forward()
    start_criterion = int(y_pred.shape[0]/2)
    loss = criterion(y_pred[start_criterion:], y_true[start_criterion:])

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
#    bucket_net0.set_value(PRECIP_RECORD, torch.tensor(network_precip_input_list, requires_grad=False))
    bucket_net0.set_value(PRECIP_RECORD, torch.tensor(network_precip_input_list, requires_grad=True))
    bucket_net0.summarize_network()
    inital_mass_in_network0 = torch.sum(torch.tensor([tensor.item() for tensor in bucket_net0.sum_H_per_layer]))
    network_outflow_tensor_0 = bucket_net0.forward()
    final_mass_in_network0 = torch.sum(torch.stack(bucket_net0.sum_H_per_layer))
    bucket_net0.report_out_mass_balance()

    ###########################################################################
    ###########################################################################
    # Train theta values of Network1
    cfg_file="./config_1_fcnn.json"
    bucket_net1 = NashCascadeNetwork(cfg_file=cfg_file)
    bucket_net1.initialize()
#    bucket_net1.set_value(PRECIP_RECORD, torch.tensor(network_precip_input_list, requires_grad=False))
    bucket_net1.set_value(PRECIP_RECORD, torch.tensor(network_precip_input_list, requires_grad=True))
    bucket_net1.unit_precip = unit_precip
    y_pred, loss = train_model(bucket_net1, network_precip_tensor.detach(), network_outflow_tensor_0.detach())
    bucket_net1.report_out_mass_balance()
    if N_TIMESTEPS <= 5:
        from torchviz import make_dot
        make_dot(loss).view()