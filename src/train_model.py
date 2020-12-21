
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F



# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def load_data(DATA_PATH, batch_size):
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH, train=False, transform=trans)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, device):
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Run the forward pass
            outputs = model(images)
            loss = F.nll_loss(outputs, labels)

            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
    return loss_list


def test_model(model, test_loader, device, MODEL_STORE_PATH):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(
            (correct / total) * 100))

    # Save the model and plot
    torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.pt')

def print_loss(loss_list):
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # GPU Device
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Hyperparameters
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001

    DATA_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/data/'
    MODEL_STORE_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/model/'


    train_loader, test_loader = load_data(DATA_PATH, batch_size)
    model = ConvNet().to(device)

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    loss_list = train_model(model, train_loader, device)


    # Load Model:
    # model = ConvNet().to(device)
    # model.load_state_dict(torch.load(MODEL_STORE_PATH+"conv_net_model.pt"))


    # Evaluate Model:
    model.eval()
    test_model(model, test_loader, device, MODEL_STORE_PATH)

    print_loss(loss_list)