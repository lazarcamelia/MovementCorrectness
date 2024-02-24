from FitGenerator import custom_fit_generator
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed

NR_FEATURES = 66

class LSTMExperiments:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.fit_generator = custom_fit_generator
        self.model = None

    def run_experiment(self):
        input_shape = (None, NR_FEATURES)  # None for variable timestamp size
        self.create_lstm_model(input_shape)

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.fit_generator(self.data, self.labels), epochs=10, steps_per_epoch=len(self.data))

    def create_lstm_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(128, input_shape=input_shape, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))  # Output layer for binary classification
        self.model = model
