import torch

from DataFrameGenerator import DataFrameGenerator
from PythonASTTreeBasedStructureGenerator import PythonASTTreeBasedStructureGenerator as tree_generator
from RecursiveBigPythonFileGenerator import RecursiveBigPythonFileGenerator as BigFileCreator

import pandas as pd
import numpy as np

file_path_to_export_keywords_from = "created_file.py"

# BigFileCreator("./codes/func_vs_proc", file_path_to_export_keywords_from).create(limit=50)

tree_for_ast_element_detection = tree_generator().from_file(file_path_to_export_keywords_from).generate()

df: pd.DataFrame = DataFrameGenerator.generate_dummy(tree_for_ast_element_detection, feature_vector_size=5, save=True)
