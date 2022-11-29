import tensorflow as tf
from keras.layers import Dense, Bidirectional, LSTM, GRU


class BaselineLSTM(tf.keras.Model):
    def __init__(self, num_classes=45):
        super().__init__()
        self.lstm = Bidirectional(LSTM(units=64, activation='relu', return_sequences=True))
        self.dense = Dense(units=num_classes, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense(x)
        return x


class AdditionalLSTM(tf.keras.Model):
    def __init__(self, num_classes=45):
        super().__init__()
        self.lstm1 = Bidirectional(LSTM(units=64, activation='relu', return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(units=64, activation='relu', return_sequences=True))
        self.dense = Dense(units=num_classes, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dense(x)
        return x


class AdditionalDense(tf.keras.Model):
    def __init__(self, num_classes=45):
        super().__init__()
        self.lstm = Bidirectional(LSTM(units=64, activation='relu', return_sequences=True))
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=num_classes, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class GRUModel(tf.keras.Model):
    def __init__(self, num_classes=45):
        super().__init__()
        self.gru = GRU(units=64, activation='relu', return_sequences=True)
        self.dense = Dense(units=num_classes, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.dense(x)
        return x
