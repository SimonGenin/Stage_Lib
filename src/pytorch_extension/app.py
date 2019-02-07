import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timeit import default_timer as timer

from tensorboardX import SummaryWriter

from TreeBasedConvolutionnalNeuralNetwork import TreeBasedConvolutionnalNeuralNetwork as tbcnn

from embedding_loaders import load_embeddings_df, load_simpler_one_hot_encoded_embedding_df, load_dummy_embedding_df
from source_code_loaders import load_source_codes


def generate_kernels(f, d):
    for i in range(1, f):
        for j in range(1, d):
            yield (i, j)


if __name__ == '__main__':

    # Prepare tensorboard elements
    tensorboard = SummaryWriter()

    # Get the python embedding vectors
    start = timer()
    embeddings_df, number_of_features_per_vector = load_dummy_embedding_df('python_stage')
    end = timer()
    print("Embeddings loaded in", end - start)

    # How many programs are going to be used to train our model ?
    number_of_programs_for_training = 2

    # How many epochs do we want ?
    number_of_epochs = 50

    # Our beautiful neural network
    net = tbcnn(number_of_features_per_vector, kernels=[(2, 2)], linear_output=2)
    soft = nn.Softmax(dim=0)

    # How are we going to train our nn ?
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=3)

    # Some global variables for logging
    loss = None

    # Load the data and tree structures we need
    start = timer()
    loaded_procedure_programs_data, \
    loaded_procedure_programs_tree, \
    loaded_function_programs_data, \
    loaded_function_programs_tree = load_source_codes(number_of_programs_for_training, embeddings_df)
    end = timer()
    print("Data & trees loaded in", end - start)

    y_expected_for_procedures = torch.tensor([1, 0], dtype=torch.float)
    y_expected_for_functions = torch.tensor([0, 1], dtype=torch.float)


    net.train()

    for current_epoch in range(number_of_epochs):

        start_epoch = timer()

        for current_program in range(number_of_programs_for_training):

            # Learn a procedure
            data = loaded_procedure_programs_data[current_program]

            y_pred = net(data, loaded_procedure_programs_tree[current_program])

            print(y_pred)

            loss = criterion(y_pred, y_expected_for_procedures)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Give some feedback on the epoch
        print()
        print("Half Epoch, after proc :", current_epoch)
        print("Loss  :", loss.item())
        print()

        for current_program in range(number_of_programs_for_training):

            # Learn a function
            data = loaded_function_programs_data[current_program]

            y_pred = net(data, loaded_function_programs_tree[current_program])

            print(y_pred)

            loss = criterion(y_pred, y_expected_for_functions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_epoch = timer()

        # Give some feedback on the epoch
        print()
        print("Half Epoch, after func :", current_epoch)
        print("Loss  :", loss.item())
        print("Time  :", str(datetime.timedelta(seconds=int(end_epoch - start_epoch))))
        print()

        # if loss < 0.001:
        #     print("Loss is less than 0.001, early stopping the net")
        #     break

    timestamp = datetime.datetime.now().isoformat()
    # torch.save(net, 'tbcnn_' + str(timestamp) + '.pt')
    torch.save(net, 'tbcnn_keep.pt')
