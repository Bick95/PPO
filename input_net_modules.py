import torch.nn as nn
import torch.nn.functional as F


class CNN_Module(nn.Module):

    def __init__(self):
        super(CNN_Module, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=3,    # 3 color channels
                                out_channels=16,  # 16 output channels = 16 filters
                                kernel_size=8,    # 8x8 kernel/filter size
                                stride=4
                                )
        self.conv_2 = nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=4,
                                stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 4 * 4)  # Flatten
        return x


# TODO: add MLP input module here



