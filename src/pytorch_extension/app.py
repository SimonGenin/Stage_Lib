import torch
import torch.nn as nn

from treelib import Tree

from pytorch_extension.TreeBasedConvolutionnalNeuralNetwork import MyNet
from torchviz import make_dot, make_dot_from_trace

from pytorch_extension.TreeBaseConvolutionLayer import TreeBasedConvolutionLayer as tbcl
from pytorch_extension.TreeBasedMaxPoolingLayer import TreeBasedMaxPoolingLayer as tbmpl

from tensorboardX import SummaryWriter


if __name__ == '__main__':
    root_id = 0
    left_id = 1
    middle_id = 2
    right_id = 3
    bottom_left_id = 4
    bottom_right_id = 5

    features = 30
    N = 20

    tree = Tree()
    tree.create_node(data=root_id, identifier=root_id)
    tree.create_node(data=left_id, identifier=left_id, parent=root_id)
    tree.create_node(data=middle_id, identifier=middle_id, parent=root_id)
    tree.create_node(data=right_id, identifier=right_id, parent=root_id)
    tree.create_node(data=bottom_left_id, identifier=bottom_left_id, parent=left_id)
    tree.create_node(data=bottom_right_id, identifier=bottom_right_id, parent=left_id)

    data = torch.randn(N, features, dtype=torch.float, requires_grad=True)

    target = torch.randn(2) * 300

    net = MyNet(tree, N, features)
    conv = tbcl(tree, N, features, ([3, 2], [2, 2], [4, 2]))
    pool = tbmpl()
    soft = nn.Softmax(dim=0)
    linear = nn.Linear(3, 2)

    summary = SummaryWriter()
    summary.add_graph(net, data)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

    loss = None

    y_pred = None


    for t in range(1000000):

        y_pred = net(data)

        loss = criterion(y_pred, target)

        if (t % 100 == 0):
            print("Loss ", t, loss.item())
            summary.add_scalar("loss", loss, t)

        if loss < 0.001: break;

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print("final loss:", loss.item())
    print(y_pred)
    summary.close()

    # make_dot(net.conv(data), params=dict(net.conv.named_parameters()))

