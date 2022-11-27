import numpy as np
import tensorflow as tf
import gensim
import gensim.downloader as gloader

from collections import OrderedDict
from typing import List, Dict
from tqdm import tqdm


class KerasTokenizer():
    """
    A simple high-level wrapper for the Keras tokenizer.
    """

    def __init__(self, add_oov_terms=False, build_embedding_matrix=False, embedding_dimension=None,
                 embedding_model_type=None, tokenizer_args=None):
        if build_embedding_matrix:
            assert embedding_model_type is not None
            assert embedding_dimension is not None and type(
                embedding_dimension) == int

        self.add_oov_terms = add_oov_terms
        self.build_embedding_matrix = build_embedding_matrix
        self.embedding_dimension = embedding_dimension
        self.embedding_model_type = embedding_model_type

        self.load_embedding = True
        self.embedding_model = None
        self.embedding_matrix = None
        self.vocab = None

        tokenizer_args = {} if tokenizer_args is None else tokenizer_args
        assert isinstance(tokenizer_args, dict) or isinstance(
            tokenizer_args, OrderedDict)
        self.tokenizer_args = tokenizer_args

    def set_OVV_terms(self, oov_terms):
        self.oov_terms = oov_terms

    def build_vocab(self, data, **kwargs):
        print('Fitting tokenizer...')
        # self.tokenizer = tf.keras.layers.TextVectorization(**self.tokenizer_args)
        # self.tokenizer.adapt(data)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)
        self.tokenizer.fit_on_texts(data)
        print('Fit completed!')

        self.vocab = self.tokenizer.word_index
        # self.vocab = {key: i for i, key in enumerate(self.tokenizer.get_vocabulary())}

        if self.build_embedding_matrix:
            if self.load_embedding:
                print('Loading embedding model! It may take a while...')
                self.embedding_model = load_embedding_model(model_type=self.embedding_model_type,
                                                            embedding_dimension=self.embedding_dimension)
                self.load_embedding = False

            print('Checking OOV terms...')
            self.oov_terms = check_OOV_terms(embedding_model=self.embedding_model,
                                             word_listing=list(self.vocab.keys()))

            print('Building the embedding matrix...')
            self.embedding_matrix = build_embedding_matrix(embedding_model=self.embedding_model,
                                                           word_to_idx=self.vocab,
                                                           vocab_size=len(
                                                               self.vocab) + 1,
                                                           embedding_dimension=self.embedding_dimension,
                                                           oov_terms=self.oov_terms,
                                                           add_oov_terms=self.add_oov_terms)
            print('Done!')

    def get_info(self):
        return {
            'build_embedding_matrix': self.build_embedding_matrix,
            'embedding_dimension': self.embedding_dimension,
            'embedding_model_type': self.embedding_model_type,
            'embedding_matrix': self.embedding_matrix.shape if self.embedding_matrix is not None else self.embedding_matrix,
            'embedding_model': self.embedding_model,
            'vocab_size': len(self.vocab) + 1,
        }

    def tokenize(self, text):
        return text

    def convert_tokens_to_ids(self, tokens):
        if type(tokens) == str:
            return self.tokenizer.texts_to_sequences([tokens])[0]
        else:
            return self.tokenizer.texts_to_sequences(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.sequences_to_texts(ids)


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


def build_embedding_matrix(embedding_model: gensim.models.keyedvectors.KeyedVectors,
                           embedding_dimension: int,
                           word_to_idx: Dict[str, int],
                           vocab_size: int,
                           oov_terms: List[str],
                           add_oov_terms=False) -> np.ndarray:
    """
    Builds the embedding matrix of a specific dataset given a pre-trained word embedding model

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_to_idx: vocabulary map (word -> index) (dict)
    :param vocab_size: size of the vocabulary
    :param oov_terms: list of OOV terms (list)

    :return
        - embedding matrix that assigns a high dimensional vector to each word in the dataset specific vocabulary (shape |V| x d)
    """
    embedding_matrix = np.zeros(
        (vocab_size, embedding_dimension), dtype=np.float32)
    for word, idx in tqdm(word_to_idx.items()):
       # try:
        #     embedding_vector = embedding_model[word]
        # except (KeyError, TypeError):
        #     embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)
        #     if add_oov_terms:
        #         embedding_model.__setitem__(word, embedding_vector)
        if word in oov_terms:
            embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)
            if add_oov_terms:
                embedding_model.__setitem__(word, embedding_vector)
        else:
            embedding_vector = embedding_model[word]

        embedding_matrix[idx] = embedding_vector

    return embedding_matrix


def convert_text(texts, tokenizer, is_training=False, max_seq_length=None):
    """
    Converts input text sequences using a given tokenizer

    :param texts: either a list or numpy ndarray of strings
    :tokenizer: an instantiated tokenizer
    :is_training: whether input texts are from the training split or not
    :max_seq_length: the max token sequence previously computed with
    training texts.

    :return
        text_ids: a nested list on token indices
        max_seq_length: the max token sequence previously computed with
        training texts.
    """
    text_ids = tokenizer.convert_tokens_to_ids(texts)

    # Padding
    if is_training:
        max_seq_length = int(np.quantile([len(seq) for seq in text_ids], 0.99))
    else:
        assert max_seq_length is not None

    text_ids = [seq + [0] * (max_seq_length - len(seq)) for seq in text_ids]
    text_ids = np.array([seq[:max_seq_length] for seq in text_ids])

    if is_training:
        return text_ids, max_seq_length
    else:
        return text_ids
