import numpy as np
import pandas as pd


class DataFrameGenerator:

    @staticmethod
    def generate_one_hot_encoded(tree, save=False, filename="one_hot_encoded.csv"):

        nodes_seen = []
        occurrences = {}

        for node in tree.all_nodes():
            if node.data in nodes_seen:
                occurrences[node.data] += 1
            else:
                nodes_seen.append(node.data)
                occurrences[node.data] = 1

        arr = np.eye(len(nodes_seen))

        df = pd.DataFrame(arr, nodes_seen, dtype=int)

        if save:
            df.to_csv(filename)

        return df

    @staticmethod
    def generate_dummy(tree, feature_vector_size=5, save=False, filename="dummy_encoding.csv"):

        nodes_seen = []
        occurrences = {}

        for node in tree.all_nodes():
            if node.data in nodes_seen:
                occurrences[node.data] += 1
            else:
                nodes_seen.append(node.data)
                occurrences[node.data] = 1

        arr = np.random.randint(10, size=(len(nodes_seen), feature_vector_size))

        df = pd.DataFrame(arr, nodes_seen, dtype=int)

        if save:
            df.to_csv(filename)

        return df
