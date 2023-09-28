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

G = 9.81
PRECIP_SVN = "atmosphere_water__liquid_equivalent_precipitation_rate"

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
        self.learning_rate = cfg_loaded['learning_rate']
        self.epochs = cfg_loaded['epochs']
        torch.cuda.manual_seed(cfg_loaded['seed'])
        torch.manual_seed(cfg_loaded['seed'])
        if cfg_loaded['verbose'] == "True":
            self.verbose = True
        else:
            self.verbose = False

    # ___________________________________________________
    ## SANITY CHECK ON GRADIENT VALUE
    def check_the_gradient_value_on_theta(self, function_str):
        print(f"WARNING: Checking Gradients {function_str}: self.theta.grad", self.ncn.theta.grad)
        # if not self.theta.grad:
        #     print(f"WARNING: Checking Gradients {function_str}: self.theta.grad", self.theta.grad)

    # ________________________________________________
    # Forward 
    def forward(self, u):
        N_TIMESTEPS = u.shape[0]

        y_pred = self.ncn.run_ncn_sequence(u)

        return y_pred


# ___________________________________________________
## PARAMETER TUNING
def train_theta_values(model, cfg_file, u, y_true):
    """ This function is used to update the theta values to minimize the loss function
        Args:
            model (ncnn): NashCascadeNeuralNetwork
            u (tensor): precipitation input
            y_true (tensor): true predictions

    """

    optim = torch.optim.SGD([model.ncn.theta],lr=model.learning_rate)

    for epoch in range(model.epochs):

        if epoch > 0:
            model.ncn = NashCascadeNetwork(cfg_file=cfg_file)
            model.ncn.initialize()
            model.ncn.theta = local_theta_values

        optim.zero_grad()

        model.ncn.initialize_bucket_head_level()

        y_pred = model(u)

        err = (y_true - y_pred)

        loss = err.pow(2.0).mean() # mean squared error

        loss.backward() # run backpropagation

        optim.step() # update the parameters, just like w -= learning_rate * w.grad

        if model.verbose:
            model.check_the_gradient_value_on_theta("After the Backwards step")
            print(f"loss is: {loss:.1f}, theta[0][0][0] is: {model.ncn.theta[0][0][0]:.1f}")
            model.ncn.report_out_mass_balance()
        else:
            print(f"loss: {loss:.1f}")

#        model.ncn.detach_ncn_from_graph()
        local_theta_values = model.ncn.theta#.detach()
        u = u.detach()
        y_true = y_true.detach()

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
    N_TIMESTEPS = 5
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

    ###########################################################################
    ###########################################################################
    # Train theta values of Network1
    cfg_file="./config_1.json"
    bucket_net1 = NashCascadeNeuralNetwork(cfg_file=cfg_file)
    loss = train_theta_values(bucket_net1, cfg_file, network_precip_tensor, network_outflow_tensor_0)
    from torchviz import make_dot
    make_dot(loss).view()
    loss.detach()