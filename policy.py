import gym
from input_net_modules import InMLP, InCNN
from output_net_modules import OutMLP
from constants import DISCRETE, CONTINUOUS

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class Policy(nn.Module):

    def __init__(self,
                 action_space: gym.spaces.Discrete or gym.spaces.Box,
                 observation_space: gym.Space,
                 input_net_type: str = 'CNN'
                 ):

        super(Policy, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.dist_type = DISCRETE if isinstance(self.action_space, gym.spaces.Discrete) else CONTINUOUS
        # NOTE!
        # In Continuous case, x actions may be sampled concurrently per env.
        # In Discrete case, only one action is sampled at a time from action space in a given env.
        # Thus, meaning of self.num_actions varies between these two output spaces!
        self.num_actions = self.action_space.n if self.dist_type is CONTINUOUS else len(action_space.sample())

        if input_net_type.lower() == 'cnn' or input_net_type.lower() == 'visual':
            # Create CNN-NN to encode inputs
            self.input_module = None  # TODO: InCNN

        else:
            # Compute nr of input features for given gym env
            input_features = sum(self.observation_space.sample().shape)

            # Create MLP-NN to encode inputs
            self.input_module = InMLP(input_features)

        self.output_module = OutMLP(hidden_features=50,
                                    output_features=self.num_actions,
                                    output_type=self.dist_type
                                    )

        self.prob_dist = Categorical if self.dist_type is DISCRETE else Normal

        # Construct deterministic processing pipeline in policy net
        self.pipeline = [
            self.input_module,
            self.output_module
        ]

        if self.dist_type is DISCRETE:
            self.pipeline.append(torch.softmax)

        # Vars for later
        self.dist = None


    def forward(self, x: torch.tensor):

        for layer in self.pipeline:
            x = layer(x)

        if self.dist_type is DISCRETE:
            self.dist = self.prob_dist(probs=x)
        else:
            self.dist = self.prob_dist(loc=x, scale=torch.ones(self.num_actions))

        action = self.dist.sample()

        return action


    def log_probs(self, action):
        return self.dist.log_prob(action)


    def entropy(self):
        return self.dist.entropy()


    def get_non_output_layers(self):
        return self.input_module
