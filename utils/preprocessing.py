import numpy as np
import tensorflow as tf
import gensim
import gensim.downloader as gloader

from collections import OrderedDict
from typing import List, Dict
from tqdm import tqdm


def load_embedding_model(model_type: str,
                         embedding_dimension: int = 50) -> gensim.models.keyedvectors.KeyedVectors:
    """
    Loads a pre-trained word embedding model via gensim library.

    :param model_type: name of the word embedding model to load.
    :param embedding_dimension: size of the embedding space to consider

    :return
        - pre-trained word embedding model (gensim KeyedVectors object)
    """
    download_path = ""
    if model_type.strip().lower() == 'word2vec':
        download_path = "word2vec-google-news-300"

    elif model_type.strip().lower() == 'glove':
        download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)
    elif model_type.strip().lower() == 'fasttext':
        download_path = "fasttext-wiki-news-subwords-300"
    else:
        raise AttributeError(
            "Unsupported embedding model type! Available ones: word2vec, glove, fasttext")

    try:
        emb_model = gloader.load(download_path)
    except ValueError as e:
        print("Invalid embedding model name! Check the embedding dimension:")
        print("Word2Vec: 300")
        print("Glove: 50, 100, 200, 300")
        print('FastText: 300')
        raise e

    return emb_model


def check_OOV_terms(embedding_model: gensim.models.keyedvectors.KeyedVectors,
                    word_listing: List[str]):
    """
    Checks differences between pre-trained embedding model vocabulary
    and dataset specific vocabulary in order to highlight out-of-vocabulary terms.

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_listing: dataset specific vocabulary (list)

    :return
        - list of OOV terms
    """
    embedding_vocabulary = set(embedding_model.index_to_key)
    oov = set(word_listing).difference(embedding_vocabulary)
    return list(oov)


def get_OOV_embedding(embedding_model, word, size):
    """For now just a random vector, can be changed to a more sophisticated method.


    Parameters
    ----------
    embedding_model : gensim.models.keyedvectors.KeyedVectors
        pre-trained word embedding model (gensim wrapper).
    word : string
        word of which to calculate the embedding.
    size : int
        size of the embedding.

    Returns
    -------
    np.array
        of shape (size,)
    """
    return np.random.uniform(low=-0.05, high=0.05, size=size)


def add_OOV_embeddings(embedding_model, words, size):
    """Check if the given words are in the embedding model. If not, they are added with their embedding.

    Parameters
    ----------
    embedding_model : gensim.models.keyedvectors.KeyedVectors
        pre-trained word embedding model (gensim wrapper).
    words : list of str
        list of words. It can also be a list of lists of strings.
    size : int
        size of the embedding.
    """
    words_copy = words

    # if words is a list of lists, flatten it
    if hasattr(words[0], '__iter__') and type(words[0]) is not str:
        words_copy = [word for sentence in words for word in sentence]
    oov_terms = check_OOV_terms(embedding_model, words_copy)
    for word in tqdm(oov_terms):
        embedding_vector = get_OOV_embedding(embedding_model=embedding_model, word=word, size=size)
        embedding_model.__setitem__(word, embedding_vector)
