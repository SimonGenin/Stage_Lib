from torch import nn
import torch
from timeit import default_timer as timer

from TreeBaseConvolutionLayer import TreeBasedConvolutionLayer as tbcl
from TreeBasedMaxPoolingLayer import TreeBasedMaxPoolingLayer as tbmpl


class TreeBasedConvolutionnalNeuralNetwork(nn.Module):

    def __init__(self, features, kernels, linear_output):
        super(TreeBasedConvolutionnalNeuralNetwork, self).__init__()
        self.conv = tbcl(features, kernels)
        self.pool = tbmpl()
        self.linear = nn.Linear(self.conv.layer_dimension, linear_output)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, tree_model):
        y_pred = self.conv(x, tree_model)

        y_pred = self.pool(y_pred)

        y_pred = self.linear(y_pred)

        # If we use the CrossEntropyLoss, we don't use the softmax.
        # There's a LofSoftmax in the Cross entropy.
        y_pred = self.softmax(y_pred)

        return y_pred
