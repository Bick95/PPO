import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Add RNN input module


class InCNN(nn.Module):

    def __init__(self):
        super(InCNN, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=3,    # 3 color channels
                                out_channels=16,  # 16 output channels = 16 filters
                                kernel_size=8,    # 8x8 kernel/filter size
                                stride=4
                                )
        self.conv_2 = nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=4,
                                stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 4 * 4)  # Flatten
        x = F.relu(self.fc1(x))
        return x


class InMLP(nn.Module):
    def __init__(self,
                 input_features: int,
                 hidden_nodes: int or list = [50, 50, 50],
                 nonlinearity: torch.nn.functional = F.relu
                 ):
        super(InMLP, self).__init__()

        # Construct NN-processing pipeline consisting of concatenation of layers to be applied to any input
        self.pipeline = [

            # Add input layer
            nn.Linear(
                input_features,
                hidden_nodes[0] if isinstance(hidden_nodes, list) else hidden_nodes
            )

        ]

        # Add optional hidden layers
        if isinstance(hidden_nodes, list):
            for i in range(1, len(hidden_nodes)):
                self.pipeline.append(
                    nn.Linear(hidden_nodes[i-1], hidden_nodes[i])
                )

        # Register all layers
        for i, layer in enumerate(self.pipeline):
            self.add_module("layer_mlp_in_" + str(i), layer)

        self.nonlinearity = nonlinearity

    def forward(self, x):

        for layer in self.pipeline:
            x = self.nonlinearity(layer(x))

        return x


# Testing
#import gym
#env = gym.make('CartPole-v1')
#observation = env.observation_space.sample()
#observation_shape = torch.tensor(observation.shape)
#net = MLP_Module(observation_shape[0])
#net(torch.tensor(observation))
