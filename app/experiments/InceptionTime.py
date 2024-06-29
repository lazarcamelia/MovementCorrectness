# resnet model
import keras
import numpy as np
import time
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.metrics.metrics_utils import confusion_matrix
from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns




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

        learning_rate = logs['learning_rate']

        if display:
            print(f'Epoch {epoch+1:02d}: Train Loss={logs["loss"]:.4f}, Train Acc={logs["accuracy"]:.4f}, Val Loss={logs["val_loss"]:.4f}, Val Acc={logs["val_accuracy"]:.4f}, Learning rate={learning_rate:.4f}')
        else:
            print(f'Epoch {epoch+1:02d}: Loss={logs["loss"]:.4f}, Acc={logs["accuracy"]:.4f}')


class Classifier_INCEPTION:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = "/Users/camelialazar/Desktop/Master/Disertatie/ProiectNou/MovementCorrectness/app/model/"

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            # self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, verbose=1,
                                                      min_lr=0.0001)

        metrics_callback = MetricsCallback()
        # Define the early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',  # Metric to monitor
            patience=50,  # Number of epochs with no improvement after which training will be stopped
            verbose=1,  # Print messages when stopping
            restore_best_weights=True  # Restore the best model weights after stopping
        )

        # file_path = self.output_directory + 'best_model.hdf5'

        # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        #                                                    save_best_only=True, save_weights_only=True)

        # self.callbacks = [reduce_lr, model_checkpoint]
        self.callbacks = [reduce_lr, metrics_callback]

        return model


    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test, y_true, plot_test_acc=True):
        # if self.batch_size is None:
        #     mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        # else:
        #     mini_batch_size = self.batch_size

        mini_batch_size = 16

        start_time = time.time()

        if plot_test_acc:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks
                                  )
            self.plot_metrics(self.callbacks[1])

        else:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, callbacks=self.callbacks,
                                  validation_freq=10)
            self.plot_metrics(self.callbacks[1])

        duration = time.time() - start_time

        # self.plot_metrics(self.callbacks[1])

        self.model.save(self.output_directory + 'last_model.h5')

        # y_pred = self.predict(x_test, y_true,
        #                       return_df_metrics=True)

        y_pred = self.model.predict(x_test)

        # Convert probabilities to class labels
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Generate the confusion matrix
        cm = confusion_matrix(y_true, y_pred, num_classes=2)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        dict = self.model.evaluate(x_test, y_test, return_dict=True)
        print(f'Test results: {dict}')
        # print(f'Test accuracy: {accuracy}')

        #
        # # convert the predicted from binary to integer
        # y_pred = np.argmax(y_pred, axis=1)
        #
        # df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration,
        #                        plot_test_acc=plot_test_acc)
        #
        # keras.backend.clear_session()

        # return df_metrics

    def predict(self, x_test, y_true, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred

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

