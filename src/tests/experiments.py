import torch
import torch.nn as nn
import numpy as np
from timeit import default_timer as timer
import asyncio
import time
from datetime import datetime
from PythonASTTreeBasedStructureGenerator import PythonASTTreeBasedStructureGenerator as tree_generator
from embedding_loaders import load_embeddings_df, load_simpler_one_hot_encoded_embedding_df, load_dummy_embedding_df
from source_code_loaders import load_source_codes

embeddings_df, number_of_features_per_vector = load_dummy_embedding_df('python_stage')

net = torch.load('../pytorch_extension/tbcnn_keep.pt')

number_of_test = 2


y_expected_for_procedures = torch.tensor([1, 0], dtype=torch.float)
y_expected_for_functions = torch.tensor([0, 1], dtype=torch.float)

print(y_expected_for_procedures)
print(y_expected_for_functions)

loaded_procedure_programs_data, \
loaded_procedure_programs_tree, \
loaded_function_programs_data, \
loaded_function_programs_tree = load_source_codes(number_of_test, embeddings_df, 732)

stats = 0


print("Functions")
for n in range(number_of_test):
    result = net(loaded_function_programs_data[n], loaded_function_programs_tree[n])
    print("Give vector", result)
    value = torch.argmax(result)
    print(value)

for n in range(number_of_test):
    result = net(loaded_procedure_programs_data[n], loaded_procedure_programs_tree[n])
    print("Give vector", result)
    value = torch.argmax(result)
    print(value)


