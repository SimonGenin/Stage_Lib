from torch import nn
import torch

from TreeBaseConvolutionLayer import TreeBasedConvolutionLayer as tbcl
from TreeBasedMaxPoolingLayer import TreeBasedMaxPoolingLayer as tbmpl


class MyNet(nn.Module):

    def __init__(self, tree, N, features):
        super(MyNet, self).__init__()
        self.conv = tbcl(tree, N, features, ([1, 2], [2, 2], [3, 4]))
        self.pool = tbmpl()
        self.linear = nn.Linear(self.conv.layer_dimension, 2)

    def forward(self, x):

        y_pred = self.conv(x)

        y_pred = self.pool(y_pred)

        y_pred = self.linear(y_pred)

        return y_pred