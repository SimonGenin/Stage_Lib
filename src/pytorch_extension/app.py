import torch
from pytorch_extension.TreeBaseConvolutionLayer import TreeBasedConvolutionLayer as tbcnn
from treelib import Tree

if __name__ == '__main__':

    root_id = 0
    left_id = 1
    middle_id = 2
    right_id = 3
    bottom_left_id = 4
    bottom_right_id = 5

    features = 5
    N = 6

    tree = Tree()
    tree.create_node(data=root_id, identifier=root_id)
    tree.create_node(data=left_id, identifier=left_id, parent=root_id)
    tree.create_node(data=middle_id, identifier=middle_id, parent=root_id)
    tree.create_node(data=right_id, identifier=right_id, parent=root_id)
    tree.create_node(data=bottom_left_id, identifier=bottom_left_id, parent=left_id)
    tree.create_node(data=bottom_right_id, identifier=bottom_right_id, parent=left_id)

    data = torch.randn(N, features, dtype=torch.float)

    arg = 2
    target = torch.ones(N, arg)

    layer = tbcnn(tree, N, features, arg, 2)
    layer2 = tbcnn(tree, N, arg, 2, 2)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adamax(layer.parameters(), lr=1e-3)
    loss = None
    for t in range(1000000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = layer2(layer(data))

        # Compute and print loss
        loss = criterion(y_pred, target)
        if t % 1000 == 0:
            print(t, loss.item())

        if loss.item() < 0.0001:
            break

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("final loss:", loss.item())
    print(y_pred)

