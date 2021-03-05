import torch.nn as nn
import torch.nn.functional as F
from constants import DISCRETE, CONTINUOUS


class OutMLP(nn.Module):
    # This module implicitly assumes in Discrete Output mode that only one action is to be predicted at a time.

    def __init__(self,
                 output_features: int,
                 input_features: int = 50,
                 output_type: int = DISCRETE
                 ):
        super(OutMLP, self).__init__()

        # Construct NN-processing pipeline consisting of concatenation of layers to be applied to any input
        # (Avoids if-statements for whether to apply the Softmax in forward pass)
        self.pipeline = [

            # Add output layer
            nn.Linear(
                in_features=input_features,
                out_features=output_features
            )

        ]

        # Register all layers
        for i, layer in enumerate(self.pipeline):
            self.add_module("layer_mlp_out_" + str(i), layer)

        # Add optional normalization of outputs in case of Discrete distribution over action space
        if output_type is DISCRETE:
            self.pipeline.append(nn.Softmax(dim=1))

    def forward(self, x):

        for layer in self.pipeline:
            x = layer(x)

        return x
