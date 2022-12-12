import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import StringLookup, Embedding
from keras.layers import Dense, Bidirectional, LSTM, GRU


def embedding_layer(x, vocabulary, embedding_model, embedding_dimension, input_length):
    """Create embedding of an input of tokenized strings.

    Parameters
    ----------
    x : tf.Tensor
        input tensor.
    vocabulary : list of strings or list-like
        tokens in the vocabulary.
    embedding_model : gensim.KeyedVectors
        pre-trained word embedding model.
    embedding_dimension : int
        size of the embedding space. If glove is used, it can be 50, 100, 200, 300.
    input_length : int
        maximum input length. If the input is shorter, it is padded to ``input_length``.

    Returns
    -------
    tf.Tensor
    """

    embedding_matrix = np.zeros((len(vocabulary), embedding_dimension))
    for i, word in enumerate(vocabulary):
        embedding_matrix[i] = embedding_model.get_vector(word)

    embedding_layer = Embedding(*embedding_matrix.shape,
                                input_length=input_length,
                                weights=[embedding_matrix])

    embedding_layer.trainable = False

    return embedding_layer(x)


def get_output_category(x, tag_vocabulary):
    """Get the predicted category

    Parameters
    ----------
    x : tf.Tensor
        output of the model. Its shape is (pad_length, num_categories)
    tag_vocabulary : list of string
        vocabulary of the categories

    Returns
    -------
    list of str
    """
    categories = np.argmax(x, axis=-1)
    with tf.device("/cpu:0"):
        output_lookup_layer = StringLookup(vocabulary=tag_vocabulary, invert=True)
        return list(output_lookup_layer(categories).numpy())


def baselineLSTM(embedding_func, input_shape=(None,), num_classes=45):
    inputs = Input(input_shape)
    x = embedding_func(inputs, input_length=input_shape[0])

    x = Bidirectional(LSTM(units=64, activation='tanh', return_sequences=True))(x)
    x = Dense(units=num_classes, activation=tf.nn.softmax)(x)

    return Model(inputs, x)


def GRUModel(embedding_func, input_shape=(None,), num_classes=45):
    inputs = Input(input_shape)
    x = embedding_func(inputs, input_length=input_shape[0])

    x = GRU(units=64, activation='tanh', return_sequences=True)(x)
    x = Dense(units=num_classes, activation=tf.nn.softmax)(x)

    return Model(inputs, x)


def additionalLSTM(embedding_func, input_shape=(None,), num_classes=45):
    inputs = Input(input_shape)
    x = embedding_func(inputs, input_length=input_shape[0])

    x = Bidirectional(LSTM(units=64, activation='tanh', return_sequences=True))(x)
    x = Bidirectional(LSTM(units=64, activation='tanh', return_sequences=True))(x)
    x = Dense(units=num_classes, activation=tf.nn.softmax)(x)

    return Model(inputs, x)


def additionalDense(embedding_func, input_shape=(None,), num_classes=45):
    inputs = Input(input_shape)
    x = embedding_func(inputs, input_length=input_shape[0])

    x = Bidirectional(LSTM(units=64, activation='tanh', return_sequences=True))(x)
    x = Dense(units=64, activation=tf.nn.softmax)(x)
    x = Dense(units=num_classes, activation=tf.nn.softmax)(x)

    return Model(inputs, x)
