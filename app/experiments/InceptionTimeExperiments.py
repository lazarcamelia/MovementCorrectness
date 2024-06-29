import sklearn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from experiments.InceptionTime import Classifier_INCEPTION


class SkeletonDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class InceptionTimeExperiments:
    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.nr_epochs = 500
        self.nr_features = 120
        self.nr_classes = 2
        self.learning_rate = 0.0001
        self.output_directory = "Users/camelialazar/Desktop/Master/Disertatie/ProiectNou/MovementCorrectness/"

        # reshape the input data to be (nr_samples, nr_frames, nr_joints * nr_coordinates)
        input_shape = list(self.data.shape)
        self.data = self.data.reshape((input_shape[0], input_shape[1], input_shape[2] * input_shape[3]))
        print("Input shape after reshape: ", self.data.shape)

    def inception_time_model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2,
                                                                          random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                                          random_state=42)


        # save original y because later we will use binary
        y_true = y_test.astype(np.int64)
        enc = sklearn.preprocessing.OneHotEncoder()
        enc.fit(np.concatenate((y_train, y_test, y_val), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
        y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

        input_shape = x_train.shape[1:]

        model = Classifier_INCEPTION(self.output_directory, input_shape, self.nr_classes, verbose=False, nb_epochs=self.nr_epochs)
        model.fit(x_train, y_train, x_val, y_val, x_test, y_test, y_true)


    def print_confusion_matrix(self, val_all_labels, val_all_preds):
        cm = confusion_matrix(val_all_labels, val_all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    def print_metrics(self, train_acc, val_acc, train_loss, val_loss):
        # Plot the training and validation loss
        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        # Plot the training and validation accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()