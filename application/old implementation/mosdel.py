from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tensorflow.keras import layers, models

# split the input data for train and test
# input_data = all_exercises.unsqueeze(1)  # Add a channel dimension (1 channel)

def prepare_data():
    all_exercises_numpy = all_exercises.numpy()
    labels_numpy = labels.numpy()


    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        all_exercises_numpy, labels_numpy, test_size=0.2, random_state=42
    )


def create_model():
    # Define the CNN architecture
    # class CNNClassifier(nn.Module):
    #     def __init__(self):
    #         super(CNNClassifier, self).__init__()
    #         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
    #         self.pool = nn.MaxPool2d(kernel_size=(2, 2))
    #         self.fc1 = nn.Linear(16 * 50 * 32, 128)
    #         self.fc2 = nn.Linear(128, 2)  # num_classes is the number of classes you're classifying into

    #     def forward(self, x):
    #         x = self.pool(F.relu(self.conv1(x)))
    #         x = x.view(-1, 16 * 50 * 32)
    #         x = F.relu(self.fc1(x))
    #         x = self.fc2(x)
    #         return x

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(75, 22, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

def train():
    model = create_model()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=100, validation_data=(test_data, test_labels))


