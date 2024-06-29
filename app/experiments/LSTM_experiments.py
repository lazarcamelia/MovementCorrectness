import sklearn
from keras import Input, Model
from keras.src.layers import Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Lambda
from keras.src.metrics.metrics_utils import confusion_matrix
import tensorflow as tf
import keras
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns



from .fit_generator import custom_fit_generator
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Dropout, Bidirectional, GlobalMaxPooling1D, Reshape, Conv1D, \
    MaxPooling1D, Activation, Dot


# Define a custom callback to capture and print the metrics
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.labels = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs['loss'])
        self.train_acc.append(logs['accuracy'])
        display = False
        if "val_loss" in logs:
            self.val_loss.append(logs['val_loss'])
            display = True
        if "val_accuracy" in logs:
            self.val_acc.append(logs['val_accuracy'])

        # learning_rate = logs['learning_rate']

        # print(logs)

        if display:
            print(f'Epoch {epoch+1:02d}: Train Loss={logs["loss"]:.4f}, Train Acc={logs["accuracy"]:.4f}, Val Loss={logs["val_loss"]:.4f}, Val Acc={logs["val_accuracy"]:.4f}')
        else:
            print(f'Epoch {epoch+1:02d}: Loss={logs["loss"]:.4f}, Acc={logs["accuracy"]:.4f}')



class LSTMExperiments:
    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.fit_generator = custom_fit_generator
        self.model = None
        self.nr_epochs = 500
        self.nr_features = len(data[0][0])
        self.nr_classes = 2

        input_shape = list(self.data.shape)
        self.data = self.data.reshape((input_shape[0], input_shape[1], input_shape[2] * input_shape[3]))
        print("Input shape after reshape: ", self.data.shape)

    def run_experiment(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2,
                                                            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                          random_state=42)

        input_shape = x_train.shape[1:]
        print("Input shape; ", input_shape)

        enc = sklearn.preprocessing.OneHotEncoder()
        enc.fit(np.concatenate((y_train, y_test, y_val), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
        y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

        # self.create_lstm_model(input_shape)
        self.create_lstm_cnn_model(input_shape)
        # self.create_lstm_attention(input_shape)

        metrics_callback = MetricsCallback()

        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])

        self.model.fit(x_train, y_train, batch_size=32, epochs=self.nr_epochs,
                              verbose=True, validation_data=(x_val, y_val), callbacks=[metrics_callback])

        self.plot_metrics(metrics_callback)

        dict = self.model.evaluate(x_test, y_test, return_dict=True)
        print(f'Test results: {dict}')

        # evaluate the model on the test data
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print("Prediction shape: ", y_pred.shape)
        print("Labels shape: ", y_true.shape)

        # Generate the confusion matrix
        cm = confusion_matrix(y_true, y_pred, num_classes=2)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def create_lstm_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(256, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = TimeDistributed(Dense(128, activation='relu'))(x)
        x = TimeDistributed(BatchNormalization())(x)  # Add batch normalization here
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)  # Add batch normalization here
        outputs = Dense(self.nr_classes, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs)

        # model = Sequential()
        # model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        # model.add(Dropout(0.2))
        # model.add(TimeDistributed(Dense(64, activation='relu')))
        # model.add(Dropout(0.2))
        # model.add(Flatten())  # Add this line to flatten the output tensor
        # model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

        # self.model = model

    def create_lstm_cnn_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = TimeDistributed(Dense(64, activation='relu'))(x)
        x = Dropout(0.2)(x)
        # Flatten the output of the TimeDistributed layer to make it compatible with Conv1D
        x = Flatten()(x)
        # Reshape to add a channel dimension for Conv1D
        x = Reshape((-1, 1))(x)
        # Add the CNN layers
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = GlobalMaxPooling1D()(x)

        outputs = Dense(self.nr_classes, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs)

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

    def plot_metrics(self, metrics):
        average_acc = sum(metrics.val_acc) / len(metrics.val_acc)
        average_loss = sum(metrics.val_loss) / len(metrics.val_loss)

        print("Overall accuracy is: ", average_acc)
        print("Overall loss is: ", average_loss)

        # Plot the training and validation loss
        plt.figure(figsize=(8, 6))
        plt.plot(metrics.train_loss, label='Training Loss')
        plt.plot(metrics.val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot the training and validation accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(metrics.train_acc, label='Training Accuracy')
        plt.plot(metrics.val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

