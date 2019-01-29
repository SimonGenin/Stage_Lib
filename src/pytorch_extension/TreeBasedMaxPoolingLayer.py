import torch.nn as nn
import torch


class TreeBasedMaxPoolingLayer(nn.Module):

    def __init__(self):
        super(TreeBasedMaxPoolingLayer, self).__init__()

    def forward(self, x):
        return torch.cat(tuple([torch.max(t).reshape(1,) for t in x]))

