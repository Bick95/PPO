import torch.nn.functional as F
from input_net_modules import InMLP, InCNN
from output_net_modules import OutMLP
from constants import CONTINUOUS

import torch
import torch.nn as nn


class ValueNet(nn.Module):

    def __init__(self,
                 observation_sample: torch.tensor = None,
                 input_net_type: str = 'CNN',
                 shared_layers: torch.nn.Module = None,
                 nonlinearity: torch.nn.functional = F.relu,
                 network_structure: list = None,
                 ):

        super(ValueNet, self).__init__()

        # Add input module
        if shared_layers:
            # Share input layer with policy net
            self.input_module = shared_layers

        else:
            # Assign input layer possibly consisting of multiple internal layers;
            # Design dependent on nature of state observations, as well as on desired network structure provided
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

        # Automatically determine how many input nodes output module is gonna need to have
        input_features_output_module = self.input_module._modules[next(reversed(self.input_module._modules))].out_features

        # Add output module
        self.output_module = OutMLP(input_features=input_features_output_module,
                                    output_features=1,
                                    output_type=CONTINUOUS
                                    )


    def forward(self, x: torch.tensor):

        hidden = self.input_module(x)
        state_value = self.output_module(hidden)

        return state_value


    def get_non_output_layers(self):
        return self.input_module
