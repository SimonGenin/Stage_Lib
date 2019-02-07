import torch
import numpy as np

from PythonASTTreeBasedStructureGenerator import PythonASTTreeBasedStructureGenerator as tree_generator


def load_source_codes(number_of_each, language_embeddings, offset = 0):
    procedure_programs_data = []
    procedure_programs_tree = []
    function_programs_data  = []
    function_programs_tree  = []

    for i in range(offset, offset + number_of_each):
        function_path = "../embeddings/codes/func_vs_proc/functions/function_" + str(i) + ".py"
        procedure_path = "../embeddings/codes/func_vs_proc/procedures/procedure_" + str(i) + ".py"

        load_source_code_in_arrays(function_path, function_programs_data, function_programs_tree, language_embeddings)
        load_source_code_in_arrays(procedure_path, procedure_programs_data, procedure_programs_tree,
                                   language_embeddings)

    return procedure_programs_data, procedure_programs_tree, function_programs_data, function_programs_tree


def load_source_code_in_arrays(from_path, destination_code_data_array, destination_tree_structure_array,
                               language_embeddings):
    source_code_structure_tree = tree_generator().from_file(from_path).generate()
    source_code_structure_tree_node_number = len(source_code_structure_tree.all_nodes())
    print("Created tree has", source_code_structure_tree_node_number, "nodes")

    source_code_data_as_python_array = []

    for index, node in enumerate(source_code_structure_tree.all_nodes()):
        embedding_vector = language_embeddings.loc[node.data].values
        node.data = index
        source_code_data_as_python_array.append(embedding_vector)

    source_code_data_as_numpy_array = np.asarray(source_code_data_as_python_array)
    source_code_data_as_tensor = torch.tensor(source_code_data_as_numpy_array, dtype=torch.float)
    destination_code_data_array.append(source_code_data_as_tensor)
    destination_tree_structure_array.append(source_code_structure_tree)
