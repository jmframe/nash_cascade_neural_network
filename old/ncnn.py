#!/bin/python3
# Nash Cascade Neural Network
# A hydrologically intuitive deep learning network

# Set up a solution to a network of buckets where the number of buckets in each layer
# flows out to the buckets in the next layer
# The parameter on each bucket is the size and height of each spigot.

# Need a function that solves this individually at a single buckets
# Then a function that loops through and moves the water to the downstream buckets
import torch
import json
import torch.nn as nn
from ncn import NashCascadeNetwork
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class NashCascadeNeuralNetwork(nn.Module):

    def __init__(self, cfg_file=None, verbose=False):
        super(NashCascadeNeuralNetwork, self).__init__()
        self.cfg_file = cfg_file
        self.config_from_json()
        self.ncn = NashCascadeNetwork(cfg_file=cfg_file)
        self.ncn.initialize()

    # ___________________________________________________
    ## CONFIGURATIONS read in from json file
    def config_from_json(self):

        with open(self.cfg_file) as cfg_file:
            cfg_loaded = json.load(cfg_file)
        self.train_learning_rate = cfg_loaded['train_learning_rate']
        self.train_lr_step_size = cfg_loaded['train_lr_step_size']
        self.train_lr_gamma = cfg_loaded['train_lr_gamma']
        self.epochs = cfg_loaded['epochs']
        torch.cuda.manual_seed(cfg_loaded['seed'])
        torch.manual_seed(cfg_loaded['seed'])
        if cfg_loaded['verbose'] == "True":
            self.verbose = True
        else:
            self.verbose = False

# ___________________________________________________
## PARAMETER TUNING
def train_model(model, u, y_true):
    """ This function is used to update the theta values to minimize the loss function
        Args:
            model (ncnn): NashCascadeNeuralNetwork
            u (tensor): precipitation input
            y_true (tensor): true predictions

    """
    # Instantiate the loss function
    criterion = nn.MSELoss()
    # We will collect all parameters from both the LSTM and Linear layer for optimization
    lstm_params = model.ncn.lstm.parameters()
    linear_params = model.ncn.linear.parameters()
    # Combine all parameters into a single list
    all_params = list(lstm_params) + list(linear_params)

    # Create the optimizer instance with the combined parameter list
    optimizer = optim.SGD(all_params, lr=model.train_learning_rate)

    scheduler = StepLR(optimizer, step_size=model.train_lr_step_size, gamma=model.train_lr_gamma, verbose=True)

    for epoch in range(model.epochs):

        optimizer.zero_grad()

        # FORWARD PASS OF THE MODEL
        y_pred = model.ncn.forward(u)

        start_criterion = int(y_pred.shape[0]/2)
        loss = criterion(y_pred[start_criterion:], y_true[start_criterion:])

        loss.backward() # run backpropagation

        optimizer.step() # update the parameters, just like w -= learning_rate * w.grad

        print(f"loss: {loss:.4f}------------")
        print(f"theta: {model.ncn.theta}")
        print("model.ncn.lstm_output.grad")
        print(model.ncn.lstm_output.grad)

        model.ncn.detach_ncn_from_graph()

        model.ncn.first_lstm_prediction = True

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
    N_TIMESTEPS = 1000
    network_precip_input_list = []
    count = 0
    for i in range(N_TIMESTEPS):
        ###########################################################################
        if count == 1:
            network_precip_input_list.append(3.0)
        elif count > 39:
            network_precip_input_list.append(3.0)
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

    ###########################################################################
    ###########################################################################
    # Train theta values of Network1
    cfg_file="./config_1.json"
    bucket_net1 = NashCascadeNeuralNetwork(cfg_file=cfg_file)
    bucket_net1.ncn.initialize_theta_values()
    y_pred, loss = train_model(bucket_net1, network_precip_tensor.detach(), network_outflow_tensor_0.detach())
    bucket_net1.ncn.report_out_mass_balance()
    if N_TIMESTEPS <= 5:
        from torchviz import make_dot
        make_dot(loss).view()