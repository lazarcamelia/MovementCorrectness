import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Define a custom dataset class for the scheleton data
from torchvision.models import ResNet18_Weights, Inception_V3_Weights, ResNet50_Weights
from torchvision.transforms import transforms


class SkeletonDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Permute the dimensions of the sample
        sample = sample.permute(2, 0, 1).unsqueeze(0)  # (1, num_joints * num_coords, time_steps, 1)

        return sample, label

class InceptionExperiments:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.nr_epochs = 15000
        self.nr_features = 87
        self.nr_classes = 2
        self.learning_rate = 0.001

    def create_inception_model(self):
        print("Create InceptionV3 model")
        # Create the dataset
        dataset = SkeletonDataset(self.data, self.labels)

        # Define the input shape
        input_shape = (1, dataset.data.shape[2] * dataset.data.shape[3], dataset.data.shape[1])  # (num_joints * num_coords, time_steps, 1)

        # Create the data loader
        batch_size = 32
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Load the pre-trained Inception v3 model
        inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        inception = inception.float()  # Convert the model's weights to float32 precision

        # Freeze the pre-trained weights
        for param in inception.parameters():
            param.requires_grad = False

        # Modify the first layer to match the input channels
        inception.Conv2d_1a_3x3.conv = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, bias=False)
        # Define the new final layer
        inception.fc = nn.Linear(inception.fc.in_features, self.nr_classes)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(inception.parameters(), lr=self.learning_rate)

        acc_values = []
        loss_values = []

        # Training loop
        for epoch in range(self.nr_epochs):
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for inputs, labels in data_loader:
                input_transform = transforms.Resize((299, 299))  # Resize the input tensor to 299 x 299
                # Preprocess the inputs
                inputs = inputs.reshape(inputs.size(0), inputs.size(2), inputs.size(3), inputs.size(1))
                inputs = input_transform(inputs)
                # Forward pass
                outputs = inception(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                running_total += labels.size(0)
                running_correct += (predicted == labels).sum().item()
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loader)
            epoch_acc = running_correct / running_total

            loss_values.append(epoch_loss)
            acc_values.append(epoch_acc)

            # Print the loss for the current epoch
            print(f"Epoch [{epoch + 1}/{self.nr_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        self.print_metric("Accuracy", acc_values)
        self.print_metric("Loss", loss_values)

    def print_metric(self, train_acc, val_acc, train_loss, val_loss):
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