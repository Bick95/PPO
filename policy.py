import gym
import torch.nn.functional as F
from input_net_modules import InMLP, InCNN
from output_net_modules import OutMLP
from constants import DISCRETE, CONTINUOUS

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class Policy(nn.Module):

    def __init__(self,
                 dist_type: int,
                 action_space: gym.spaces.Discrete or gym.spaces.Box,
                 observation_sample: torch.tensor,
                 input_net_type: str = 'CNN',
                 standard_dev: float = 1.,
                 nonlinearity: torch.nn.functional = F.relu,
                 network_structure: list = None,
                 ):

        super(Policy, self).__init__()

        # Determine whether output distribution is to be Discrete or Continuous
        self.dist_type = dist_type
        # NOTE!
        # In Continuous case, n actions may be sampled concurrently per env.
        # In Discrete case, only one action (out of n options) is sampled at a time from the action space in a given env.
        # Thus, meaning of self.num_actions varies between these two output spaces/cases!
        if self.dist_type is DISCRETE:
            self.num_actions = action_space.n
        else:
            # Assumption: no flattening needed!
            self.num_actions = action_space.shape[0]

        # Assign tensor of standard deviations
        std = [standard_dev] * self.num_actions
        print(std)
        self.std = torch.tensor(std, requires_grad=False)

        # Assign input layer possibly consisting of multiple internal layers; Design dependent on nature of state observations
        if input_net_type.lower() == 'cnn' or input_net_type.lower() == 'visual':
            # Create CNN to encode inputs
            self.input_module = InCNN(
                network_structure=network_structure,
                input_sample=observation_sample,
                nonlinearity=nonlinearity,
            )

        else:
            # Create MLP-NN to encode inputs
            self.input_module = InMLP(network_structure=network_structure,
                                      input_sample=observation_sample,
                                      nonlinearity=nonlinearity,
                                      )

        # Automatically determine how many input nodes the output module/layer is gonna need to have
        input_features_output_module = self.input_module._modules[next(reversed(self.input_module._modules))].out_features

        # Assign (deterministic) output layer for generating parameterizations of probability distributions over action space to be defined below
        self.output_module = OutMLP(input_features=input_features_output_module,
                                    output_features=self.num_actions,
                                    output_type=self.dist_type
                                    )

        # Assign stochastic probability distribution (including sampler) for sampling actions
        self.prob_dist = Categorical if self.dist_type is DISCRETE else Normal

        # Construct the deterministic processing pipeline of the policy net
        self.pipeline = [
            self.input_module,
            self.output_module
        ]

        # To be assigned later (during execution). Will contain parameterized distribution (one per parallel agent)
        self.dist = None  # Will contain concrete probability distributions for sampling actions


    def forward(self, x: torch.tensor):

        for layer in self.pipeline:
            x = layer(x)

        if self.dist_type is DISCRETE:
            self.dist = self.prob_dist(probs=x)
        else:
            self.dist = self.prob_dist(loc=x, scale=self.std)

        action = self.dist.sample()

        return action


    def forward_deterministic(self, x: torch.tensor):
        # Executes a standard forward pass through the policy, but then does not sample actions, but determines them deterministically

        for layer in self.pipeline:
            x = layer(x)

        if self.dist_type is DISCRETE:
            #print("Parameterization discrete:", x)
            action = torch.argmax(x, dim=-1)  # Choose for every minibatch example the action with largest probability mass
        else:
            #print("Parameterization continuous:", x)
            action = x  # x contains the mean of a Gaussian per minibatch example. It's like sampling with 0 standard deviation

        #print("Action:", action)
        return action


    def log_prob(self, action):
        return self.dist.log_prob(action)


    def entropy(self):
        return self.dist.entropy()


    def get_non_output_layers(self):
        return self.input_module
