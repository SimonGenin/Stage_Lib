from torch import nn
import torch

from pytorch_extension.TreeBaseConvolutionLayer import TreeBasedConvolutionLayer as tbcl
from pytorch_extension.TreeBasedMaxPoolingLayer import TreeBasedMaxPoolingLayer as tbmpl


class MyNet(nn.Module):

    def __init__(self, tree, N, features):
        super(MyNet, self).__init__()
        self.conv = tbcl(tree, N, features, ([2, 5], [2, 2], [2, 2]))
        self.pool = tbmpl()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):

        y_pred = self.conv(x)

        y_pred = self.pool(y_pred)

        y_pred = self.linear(y_pred)

        return y_pred