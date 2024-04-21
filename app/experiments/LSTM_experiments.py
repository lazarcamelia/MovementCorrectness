from keras.src.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Reshape, Activation, Dot
from tensorflow.python.keras.layers import Lambda

from .fit_generator import custom_fit_generator
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Dropout, Bidirectional


class LSTMExperiments:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.fit_generator = custom_fit_generator
        self.model = None
        self.nr_epochs = 10
        self.nr_features = len(data[0][0])

    def run_experiment(self):
        input_shape = (None, self.nr_features)  # None for variable timestamp size
        self.create_lstm_model(input_shape)
        # self.create_lstm_cnn_model(input_shape)
        # self.create_lstm_attention(input_shape)

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.fit_generator(self.data, self.labels), epochs=self.nr_epochs, steps_per_epoch=len(self.data))

    def create_lstm_model(self, input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(64, activation='relu')))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

        self.model = model

    def create_lstm_cnn_model(self, input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(64, activation='relu')))
        model.add(Dropout(0.2))

        # Flatten or reshape the output of the LSTM layers
        model.add(GlobalMaxPooling1D())
        model.add(Reshape((-1, 1)))
        # Add the CNN layers
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dense(1, activation='sigmoid'))

        self.model = model

    def create_lstm_attention(self, input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(64, activation='relu')))
        model.add(Dropout(0.2))

        # Apply attention mechanism
        model.add(Lambda(self.attention))

        # Add CNN layers
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        # Flatten the output before the final dense layer
        model.add(GlobalMaxPooling1D())

        # Final dense layer for binary classification
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model

    def attention(self, inputs):
        lstm_output = inputs[0]  # Accessing the first element of the inputs tuple
        attention_weights = Dense(1, activation='tanh')(lstm_output)
        attention_weights = Activation('softmax')(attention_weights)
        context_vector = Dot(axes=1)([lstm_output, attention_weights])
        return context_vector

