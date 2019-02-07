import pandas as pd


def load_embeddings_df(language='python_stage'):
    embeddings_df = None

    if language == 'python_stage':
        embeddings_df = pd.read_csv('../embeddings/one_hot_encoded.csv')

    embeddings_df.set_index('expr', inplace=True)
    feature_vector_size = embeddings_df.shape[1]

    return embeddings_df, feature_vector_size


def load_simpler_one_hot_encoded_embedding_df(language="python_stage"):
    embeddings_df = None

    if language == 'python_stage':
        embeddings_df = pd.read_csv('../embeddings/simple_one_hot.csv')

    embeddings_df.set_index('expr', inplace=True)
    feature_vector_size = embeddings_df.shape[1]

    return embeddings_df, feature_vector_size


def load_dummy_embedding_df(language="python_stage"):
    embeddings_df = None

    if language == 'python_stage':
        embeddings_df = pd.read_csv('../embeddings/dummy_encoding.csv')

    embeddings_df.set_index('expr', inplace=True)
    feature_vector_size = embeddings_df.shape[1]

    return embeddings_df, feature_vector_size
