import torch
import torch.nn as nn


# noinspection PyMethodMayBeStatic
class TreeBasedConvolutionLayer(nn.Module):

    def __init__(self, tree_model, batch_size, feature_size, number_of_features_to_detect, depth_of_sliding_window):

        super(TreeBasedConvolutionLayer, self).__init__()

        self.tree_model = tree_model
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.number_features_detection = number_of_features_to_detect
        self.sliding_window_depth = depth_of_sliding_window

        self.layer_weight_top = nn.Parameter(torch.randn(self.number_features_detection, self.feature_size))
        self.layer_weight_right = nn.Parameter(torch.randn(self.number_features_detection, self.feature_size))
        self.layer_weight_left = nn.Parameter(torch.randn(self.number_features_detection, self.feature_size))

        self.layer_bias = nn.Parameter(torch.randn(self.number_features_detection))

    def forward(self, tree_data):

        convoluted_data = torch.zeros(self.batch_size, self.number_features_detection)

        for (index, node) in enumerate(self.tree_model.all_nodes()):

            current_window_position_node_id = node.data

            current_node_depth = self.tree_model.level(current_window_position_node_id) + 1

            hovered_nodes_by_window = [ids.data for ids in self.tree_model.children(current_window_position_node_id)]
            hovered_nodes_by_window.insert(0, current_window_position_node_id)

            summed_data = torch.zeros(self.number_features_detection)
            for n_id in hovered_nodes_by_window:
                # Prepare the coefficients for the continuous binary tree weights
                twc = self.top_weight_coef(current_node_depth)
                rwc = self.right_weight_coef(twc, self.get_siblings_number(n_id),
                                             self.get_node_position_amongst_siblings(n_id))
                lwc = self.left_weight_coef(twc, rwc)

                # Prepare the convolution weight matrix
                weight_convolution = twc * self.layer_weight_top + lwc * self.layer_weight_left + rwc * self.layer_weight_right

                # Get the sum ready for the convolution
                summed_data = summed_data + (
                        weight_convolution.mm(tree_data[n_id].unsqueeze(0).t()).t() + self.layer_bias)

            convoluted_data[current_window_position_node_id] = torch.tanh(summed_data)

        return convoluted_data

    def top_weight_coef(self, node_depth):
        return (node_depth - 1) / (self.sliding_window_depth - 1)

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
