import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Bidirectional, LSTM, GRU


def baselineLSTM(input_shape=(200, 50), num_classes=45):
    inputs = Input(input_shape)
    x = Bidirectional(LSTM(units=64, activation='relu', return_sequences=True))(inputs)
    x = Dense(units=num_classes, activation=tf.nn.softmax)(x)

    return Model(inputs, x)


def GRUModel(input_shape=(200, 50), num_classes=45):
    inputs = Input(input_shape)
    x = GRU(units=64, activation='relu', return_sequences=True)(inputs)
    x = Dense(units=num_classes, activation=tf.nn.softmax)(x)

    return Model(inputs, x)


def additionalLSTM(input_shape=(200, 50), num_classes=45):
    inputs = Input(input_shape)
    x = Bidirectional(LSTM(units=64, activation='relu', return_sequences=True))(inputs)
    x = Bidirectional(LSTM(units=64, activation='relu', return_sequences=True))(x)
    x = Dense(units=num_classes, activation=tf.nn.softmax)(x)

    return Model(inputs, x)


def additionalDense(input_shape=(200, 50), num_classes=45):
    inputs = Input(input_shape)
    x = Bidirectional(LSTM(units=64, activation='relu', return_sequences=True))(inputs)
    x = Dense(units=64, activation=tf.nn.softmax)(x)
    x = Dense(units=num_classes, activation=tf.nn.softmax)(x)

    return Model(inputs, x)
