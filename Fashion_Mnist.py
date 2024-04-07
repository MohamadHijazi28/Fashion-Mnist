import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch import optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import numpy as np
from collections import namedtuple
from torchvision.utils import make_grid
from tqdm import tqdm

# hyper parameters
input_size = 784  # 28X28
hidden_size = 60  # the hidden_size decide the parameters number
num_epochs = 50
batch_size = 1000
learning_rate = 0.001


# Sample image in dataset
def view_data_sample(loader):
    image, label = next(iter(loader))
    plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.imshow(make_grid(image, nrow=16).permute((1, 2, 0)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def splice_batch(X, Y, num_of_labels, prints=False):
    if prints:
        print('input: ', end="")
        print("\t X shape: ", X.shape, end='\t')
        print("\t Y shape: ", Y.shape)
    X = X[Y < num_of_labels]
    Y = Y[Y < num_of_labels]
    if prints:
        print('output: ', end="")
        print("\t X shape: ", X.shape, end='\t')
        print("\t Y shape: ", Y.shape)
    return X, Y


def train_test(train_data, train_targets, test_data, test_targets):
    train_data = train_data.float() / 1.0  # Normalization
    train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = test_data.float() / 1.0  # Normalization
    test_dataset = torch.utils.data.TensorDataset(test_data, test_targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def plot_results(train_losses, test_losses, train_accuracies, test_accuracies, network_index):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot train and test losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(f'Network {network_index} Error')
    plt.legend()
    plt.show()
    # Plot train and test accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train')
    plt.plot(epochs, test_accuracies, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Network {network_index} Accuracy')
    plt.legend()

    #plt.tight_layout()
    plt.show()


def train_test_epoch(model_1, train_loader, test_data, test_targets, optimizer, network_index):
    train_losses, train_accuracies, test_losses, test_accuracies = ([], [], [], [])

    for epoch in range(num_epochs):
        model_1, train_loss, train_accuracy = train_model(model_1, train_loader, optimizer)
        test_accuracy, test_loss = test_model(model_1, test_data, test_targets, network_index)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)
        print(f'Epoch {epoch}/{num_epochs}, Average Loss: {train_loss:.4f}, Accuracy: {train_accuracy * 100:.2f}%')

    return train_losses,train_accuracies,test_losses, test_accuracies


# define the first network, 3 classes{0,1,2} and 2 weighted layers
def first_network():
    train_data, train_targets = splice_batch(trainset.data, trainset.targets, 3)
    test_data, test_targets = splice_batch(testset.data, testset.targets, 3)
    train_loader, test_loader = train_test(train_data, train_targets, test_data, test_targets)

    model_1 = NeuralNetwork(3)
    optimizer = optim.Adam(model_1.parameters(), lr=learning_rate)
    train_losses,train_accuracies,test_losses, test_accuracies = train_test_epoch(model_1, train_loader,test_data, test_targets, optimizer, 1)
    plot_results(train_losses, test_losses, train_accuracies, test_accuracies, 1)
    torch.save(model_1.state_dict(), 'model_1.pth')  # Save the trained model
    return model_1


def load_first_network():
    model_1 = NeuralNetwork(3)
    model_1.load_state_dict(torch.load('model_1.pth'))  # Load the trained model

    return model_1


# define the second network, 7 classes{0,1,2,3,4,5,6} and 2 weighted layers
def second_network():
    train_data, train_targets = splice_batch(trainset.data, trainset.targets, 7)
    test_data, test_targets = splice_batch(testset.data, testset.targets, 7)
    train_loader, test_loader = train_test(train_data, train_targets, test_data, test_targets)

    model_2 = NeuralNetwork(7)
    optimizer = optim.Adam(model_2.parameters(), lr=learning_rate)
    train_losses, train_accuracies, test_losses, test_accuracies = train_test_epoch(model_2, train_loader, test_data, test_targets, optimizer, 2)
    plot_results(train_losses, test_losses, train_accuracies, test_accuracies, 2)
    torch.save(model_2.state_dict(), 'model_2.pth')  # Save the trained model
    return model_2


def load_second_network():
    model_2 = NeuralNetwork(7)
    model_2.load_state_dict(torch.load('model_2.pth'))  # Load the trained model
    return model_2


# define the third network, 7 classes{0,1,2,3,4,5,6} and 4 weighted layers
def third_network():
    train_data, train_targets = splice_batch(trainset.data, trainset.targets, 7)
    test_data, test_targets = splice_batch(testset.data, testset.targets, 7)
    train_loader, test_loader = train_test(train_data, train_targets, test_data, test_targets)

    model_3 = NeuralNetwork4weighted(7)
    optimizer = optim.Adam(model_3.parameters(), lr=learning_rate)
    train_losses, train_accuracies, test_losses, test_accuracies = train_test_epoch(model_3, train_loader, test_data, test_targets, optimizer, 3)
    plot_results(train_losses, test_losses, train_accuracies, test_accuracies, 3)
    torch.save(model_3.state_dict(), 'model_3.pth')  # Save the trained model
    return model_3


def load_third_network():
    model_3 = NeuralNetwork4weighted(7)
    model_3.load_state_dict(torch.load('model_3.pth'))  # Load the trained model
    return model_3


# define the fourth network, 7 classes{0,1,2,3,4,5,6} and 4 weighted layers
def fourth_network():
    train_data, train_targets = splice_batch(trainset.data, trainset.targets, 7)
    test_data, test_targets = splice_batch(testset.data, testset.targets, 7)
    train_loader, test_loader = train_test(train_data, train_targets, test_data, test_targets)

    model_4 = CNNNeuralNetwork(7)
    optimizer = optim.Adam(model_4.parameters(), lr=learning_rate)
    train_losses,train_accuracies,test_losses, test_accuracies = train_test_epoch(model_4, train_loader, test_data, test_targets, optimizer, 4)
    plot_results(train_losses, test_losses, train_accuracies, test_accuracies, 4)
    torch.save(model_4.state_dict(), 'model_4.pth')  # Save the trained model
    return model_4


def load_fourth_network():
    model_4 = CNNNeuralNetwork(7)
    model_4.load_state_dict(torch.load('model_4.pth'))  # Load the trained model
    return model_4


# Define the network architecture (2 weighted layers)
class NeuralNetwork(nn.Module):
    def __init__(self, classes_num):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)  # input_size=784(28*28), hidden_size=60
        self.l2 = nn.Linear(hidden_size, classes_num)  # classes_num is output classes = 3/7 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class NeuralNetwork4weighted(nn.Module):
    def __init__(self, classes_num):
        super(NeuralNetwork4weighted, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)  # input_size=784(28*28)
        self.l2 = nn.Linear(hidden_size, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, classes_num)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = self.l4(x)
        return x


class CNNNeuralNetwork(nn.Module):
    def __init__(self, classes_num):
        super(CNNNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 3)
        self.bn1 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(256, 128, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(128, 70, 1)
        self.bn3 = nn.BatchNorm2d(70)
        self.dropout3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(70 * 3 * 3, classes_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_model(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, labels in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(train_loader)
    accuracy = correct / total

    train_losses = average_loss
    train_accuracies = accuracy

    return model, train_losses, train_accuracies


def test_model(model, test_data, test_targets, network_index):
    test_losses = 0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        test_data = test_data.float() / 1.0  # Normalization
        test_dataset = torch.utils.data.TensorDataset(test_data, test_targets)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for data, labels in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(output, labels)
            test_losses += loss.item()
        accuracy = correct / total

    print(f'Test Accuracy for network {network_index} is: {accuracy * 100:.2f}%')
    test_losses /= len(test_loader)
    return accuracy, test_losses


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print('device: ', device)

# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()

#model_1 = first_network()
#print(f'Number of parameters in Network 1: {count_parameters(model_1)} \n')

#model_2 = second_network()
#print(f'Number of parameters in Network 2: {count_parameters(model_2)} \n')

#model_3 = third_network()
#print(f'Number of parameters in Network 3: {count_parameters(model_3)} \n')

#model_4 = fourth_network()
#print(f'Number of parameters in Network 4: {count_parameters(model_4)} \n')


model = load_first_network()
print(f'Number of parameters in Network 1: {count_parameters(model)}')
test_data, test_targets = splice_batch(testset.data, testset.targets, 3)
test_model(model, test_data, test_targets, 1)

model = load_second_network()
print(f'Number of parameters in Network 2: {count_parameters(model)}')
test_data, test_targets = splice_batch(testset.data, testset.targets, 7)
test_model(model, test_data, test_targets, 2)

model = load_third_network()
print(f'Number of parameters in Network 3: {count_parameters(model)}')
test_data, test_targets = splice_batch(testset.data, testset.targets, 7)
test_model(model, test_data, test_targets, 3)

model = load_fourth_network()
print(f'Number of parameters in Network 4: {count_parameters(model)}')
test_data, test_targets = splice_batch(testset.data, testset.targets, 7)
test_model(model, test_data, test_targets, 4)
