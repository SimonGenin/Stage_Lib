import torch
import torch.nn as nn
import numpy as np


# noinspection PyMethodMayBeStatic
class TreeBasedConvolutionLayer(nn.Module):
    """
    The kernel is the depth of the sliding window and the number of features to detect
    """

    def __init__(self, tree_model, batch_size, feature_size, kernels):

        super(TreeBasedConvolutionLayer, self).__init__()

        self.tree_model = tree_model
        self.batch_size = batch_size
        self.feature_size = feature_size

        self.kernels = kernels
        self.layer_dimension = len(self.kernels)

        number_features_detection_id = 0
        depth_of_sliding_window = 1

        self.number_features_detection = [None] * self.layer_dimension
        self.sliding_window_depth = [None] * self.layer_dimension

        self.layer_weight_top = [None] * self.layer_dimension
        self.layer_weight_right = [None] * self.layer_dimension
        self.layer_weight_left = [None] * self.layer_dimension
        self.layer_bias = [None] * self.layer_dimension

        for i in range(self.layer_dimension):
            self.number_features_detection[i] = self.kernels[i][number_features_detection_id]
            self.sliding_window_depth[i] = self.kernels[i][depth_of_sliding_window]

            self.layer_weight_top[i] = nn.Parameter(
                torch.randn(self.kernels[i][number_features_detection_id], self.feature_size))
            self.layer_weight_right[i] = nn.Parameter(
                torch.randn(self.kernels[i][number_features_detection_id], self.feature_size))
            self.layer_weight_left[i] = nn.Parameter(
                torch.randn(self.kernels[i][number_features_detection_id], self.feature_size))
            self.layer_bias[i] = nn.Parameter(torch.randn(self.kernels[i][number_features_detection_id]))

        self.params = nn.ParameterList(
            self.layer_weight_top + self.layer_weight_right + self.layer_weight_left + self.layer_bias)

    def forward_one(self, i, tree_data):
        convoluted_data = torch.zeros(self.batch_size, self.number_features_detection[i])

        for (index, node) in enumerate(self.tree_model.all_nodes()):

            current_window_position_node_id = node.data

            current_node_depth = self.tree_model.level(current_window_position_node_id) + 1

            hovered_nodes_by_window = list(self.tree_model.expand_tree(current_window_position_node_id,
                                                                  mode=self.tree_model.WIDTH,
                                                                  filter=lambda x: self.tree_model.level(
                                                                      x.identifier) <= self.tree_model.level(
                                                                      current_window_position_node_id) + self.sliding_window_depth[i] - 1))

            # print("Hovered nodes are:", hovered_nodes_by_window)

            summed_data = torch.zeros(self.number_features_detection[i])
            for n_id in hovered_nodes_by_window:
                # Prepare the coefficients for the continuous binary tree weights
                twc = self.top_weight_coef(current_node_depth, i)
                rwc = self.right_weight_coef(twc, self.get_siblings_number(n_id),
                                             self.get_node_position_amongst_siblings(n_id))
                lwc = self.left_weight_coef(twc, rwc)

                # Prepare the convolution weight matrix
                weight_convolution = twc * self.layer_weight_top[i] + lwc * self.layer_weight_left[i] + rwc * \
                                     self.layer_weight_right[i]

                # Get the sum ready for the convolution
                summed_data = summed_data + (
                        weight_convolution.mm(tree_data[n_id].unsqueeze(0).t()).t() + self.layer_bias[i])

            convoluted_data[current_window_position_node_id] = torch.tanh(summed_data)

        return convoluted_data

    def forward(self, tree_data):

        return [self.forward_one(i, tree_data) for i in range(self.layer_dimension)]

    def top_weight_coef(self, node_depth, i):
        return (node_depth - 1) / (self.sliding_window_depth[i] - 1)

    def right_weight_coef(self, top_weight_coefficient, siblings_number, node_position):
        return (1 - top_weight_coefficient) * ((node_position - 1) / max(1, siblings_number - 1))

    def left_weight_coef(self, top_weight_coefficient, right_weight_coefficient):
        return (1 - top_weight_coefficient) * (1 - right_weight_coefficient)

    def get_siblings_number(self, node_id):
        return len(self.tree_model.siblings(node_id))

    def get_node_position_amongst_siblings(self, node_id):
        if node_id == 0:
            return 1

        return next(
            index for (index, node) in enumerate(self.tree_model.children(self.tree_model.parent(node_id).data)) if
            node.data == node_id)
