import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timeit import default_timer as timer

from tensorboardX import SummaryWriter

from TreeBasedConvolutionnalNeuralNetwork import TreeBasedConvolutionnalNeuralNetwork as tbcnn

from embedding_loaders import load_embeddings_df
from source_code_loaders import load_source_codes

embeddings_df, number_of_features_per_vector = load_embeddings_df('python_stage')

programs = 5

model = torch.load('tbcnn_2019-02-04T16:40:25.781296.pt')

loaded_procedure_programs_data, \
    loaded_procedure_programs_tree, \
    loaded_function_programs_data, \
    loaded_function_programs_tree = load_source_codes(5, embeddings_df, 700)



for i in range (programs):
    result = model(loaded_procedure_programs_data[i], loaded_procedure_programs_tree[i]).reshape(1, 2)
    print(result)

print("---")

for i in range (programs):
    result = model(loaded_function_programs_data[i], loaded_function_programs_tree[i]).reshape(1, 2)
    print(result)