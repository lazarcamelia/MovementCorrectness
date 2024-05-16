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

class CNNExperiments:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.nr_epochs = 500
        self.nr_features = 87
        self.nr_classes = 2
        self.learning_rate = 0.001

    def create_inception_model(self):
        # Create the dataset
        dataset = SkeletonDataset(self.data, self.labels)

        # Define the input shape
        input_shape = (dataset.data.shape[2] * dataset.data.shape[3], dataset.data.shape[1], 1)  # (num_joints * num_coords, time_steps, 1)

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

        # Modify the spatial dimensions of the input
        inception.aux_logits = False  # Disable auxiliary classifiers
        inception_modules = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a,
                             inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e,
                             inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c]

        for module in inception_modules:
            if hasattr(module, 'branch1x1'):
                conv = module.branch1x1.conv
                conv.stride = (1, 1)

        # Define the new final layer
        num_classes = 2
        inception.fc = nn.Linear(inception.fc.in_features, num_classes)

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
                # Preprocess the inputs
                inputs = inputs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])

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

    def print_metric(self, metric_type, values_train, values_val):
        # Plot accuracy values
        x = list(range(1, len(values_train) + 1))

        training_label = "Training " + metric_type
        val_label = "Validation " + metric_type

        plt.plot(x, values_train, label=training_label, marker='o')
        plt.plot(x, values_val, label=val_label, marker='o')

        # Add labels and title
        plt.xlabel('Epochs')
        plt.ylabel(metric_type)
        plt.legend()
        plt.title(metric_type + ' Over Epochs')

        # Show grid
        plt.grid(True)

        # Show plot
        plt.show()

    def create_resnet_model(self):
        print("Training ResNet model")
        print("Nr epochs: ", self.nr_epochs)
        print("Learning rate: ", self.learning_rate)
        print("Split: 80/20")
        # Create the dataset
        dataset = SkeletonDataset(self.data, self.labels)

        # Define the input shape
        # (time_steps, num_joints, num_coords)
        input_shape = (dataset.data.shape[2] * dataset.data.shape[3], dataset.data.shape[1], 1)  # (num_joints * num_coords, time_steps, 1)

        # Create the data loader
        batch_size = 32
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Load the pre-trained ResNet model
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze the pre-trained weights
        for param in resnet.parameters():
            param.requires_grad = False

        # Modify the first layer to match the input channels
        resnet.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Define the new final layer
        num_classes = 2
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(resnet.parameters(), lr=self.learning_rate)

        acc_values = []
        loss_values = []

        # Training loop
        for epoch in range(self.nr_epochs):
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for inputs, labels in data_loader:
                # Preprocess the inputs
                inputs = inputs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])

                # Forward pass
                outputs = resnet(inputs)
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
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        self.print_metric("Accuracy", acc_values)
        self.print_metric("Loss", loss_values)

    def create_resnet_model_early_stopping(self):
        print("Training ResNet model")
        print("Nr epochs: ", self.nr_epochs)
        print("Learning rate: ", self.learning_rate)
        print("Split: 80/20")

        # Create the dataset
        train_data, val_data, train_labels, val_labels = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        train_dataset = SkeletonDataset(train_data, train_labels)
        val_dataset = SkeletonDataset(val_data, val_labels)

        # Define the input shape
        # (time_steps, num_joints, num_coords)
        input_shape = (train_dataset.data.shape[2] * train_dataset.data.shape[3], train_dataset.data.shape[1], 1)  # (num_joints * num_coords, time_steps, 1)

        # Create the data loader
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Load the pre-trained ResNet model
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # # Freeze the pre-trained weights
        # for param in resnet.parameters():
        #     param.requires_grad = False

        children = list(resnet.children())
        for child in children[:-3]:
            for param in child.parameters():
                param.requires_grad = False

        # Modify the first layer to match the input channels
        resnet.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Define the new final layer
        resnet.fc = nn.Linear(resnet.fc.in_features, self.nr_classes)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(resnet.parameters(), lr=self.learning_rate)

        patience = 20  # Number of epochs to wait before early stopping
        early_stopping_counter = 0
        best_val_loss = float('inf')

        acc_values_val = []
        loss_values_val = []

        acc_values_train = []
        loss_values_train = []

        for epoch in range(self.nr_epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # training loop
            for inputs, labels in train_loader:
                # Preprocess the inputs
                inputs = inputs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])

                # Forward pass
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item()

            training_loss = train_loss / len(train_loader)
            training_accuracy = train_correct / train_total

            acc_values_train.append(training_accuracy)
            loss_values_train.append(training_loss)

            # Validation loop
            resnet.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
                    outputs = resnet(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            acc_values_val.append(val_acc)
            loss_values_val.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Print the loss for the current epoch
            print(
                f"Epoch [{epoch + 1}/{self.nr_epochs}], Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            # print(f"Epoch [{epoch + 1}/{self.nr_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.print_metric("Accuracy", acc_values_train, acc_values_val)
        self.print_metric("Loss", loss_values_train, loss_values_val)


    def create_resnet_model_with_weighthed_loss(self):
        print("Training ResNet model")
        print("Nr epochs: ", self.nr_epochs)
        print("Learning rate: ", self.learning_rate)
        print("Split: 80/20")
        print("Weights for loss")

        # Create the dataset
        train_data, val_data, train_labels, val_labels = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        train_dataset = SkeletonDataset(train_data, train_labels)
        val_dataset = SkeletonDataset(val_data, val_labels)

        # Define the input shape
        # (time_steps, num_joints, num_coords)
        input_shape = (train_dataset.data.shape[2] * train_dataset.data.shape[3], train_dataset.data.shape[1], 1)  # (num_joints * num_coords, time_steps, 1)

        # Create the data loader
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Load the pre-trained ResNet model
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze the pre-trained weights
        for param in resnet.parameters():
            param.requires_grad = False

        # Modify the first layer to match the input channels
        resnet.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Define the new final layer
        resnet.fc = nn.Linear(resnet.fc.in_features, self.nr_classes)

        # Compute class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        # Define the loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(resnet.parameters(), lr=self.learning_rate)

        patience = 20  # Number of epochs to wait before early stopping
        early_stopping_counter = 0
        best_val_loss = float('inf')

        acc_values_val = []
        loss_values_val = []

        acc_values_train = []
        loss_values_train = []

        for epoch in range(self.nr_epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # training loop
            for inputs, labels in train_loader:
                # Preprocess the inputs
                inputs = inputs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])

                # Forward pass
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item()

            training_loss = train_loss / len(train_loader)
            training_accuracy = train_correct / train_total

            acc_values_train.append(training_accuracy)
            loss_values_train.append(training_loss)

            # Validation loop
            resnet.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
                    outputs = resnet(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            acc_values_val.append(val_acc)
            loss_values_val.append(val_loss)

            # # Early stopping
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     early_stopping_counter = 0
            # else:
            #     early_stopping_counter += 1

            # Print the loss for the current epoch
            print(
                f"Epoch [{epoch + 1}/{self.nr_epochs}], Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            # print(f"Epoch [{epoch + 1}/{self.nr_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            #
            # if early_stopping_counter >= patience:
            #     print(f"Early stopping at epoch {epoch + 1}")
            #     break

        self.print_metric("Accuracy", acc_values_train, acc_values_val)
        self.print_metric("Loss", loss_values_train, loss_values_val)