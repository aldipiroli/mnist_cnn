
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from models import *

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
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

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


def test_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            ### Show images while training:
            # print(images.cpu().shape)
            # print(predicted, outputs)
            # plt.imshow(images.cpu().view(28,28))
            # plt.show()
            # # print(images.cpu().view(28,28))
            # # cv2.imwrite(str(labels.flatten())+".png", images.cpu().view(28,28).numpy()) 
            # input()
            # plt.close('all')

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(
            (correct / total) * 100))


def save_model(model, MODEL_STORE_PATH):
    torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model_3conv_15.pt')

def print_loss(loss_list):
    plt.plot(np.arange(len(loss_list)), loss_list, 'b')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # GPU Device
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Hyperparameters
    num_epochs = 15
    batch_size = 1
    learning_rate = 0.00001

    DATA_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/data/'
    MODEL_STORE_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/model/'


    train_loader, test_loader = load_data(DATA_PATH, batch_size)
    model = ConvNet3L().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train Model:
    # loss_list = train_model(model, train_loader, device)
    # print_loss(loss_list)

    # Save Model:
    # save_model(model, MODEL_STORE_PATH)

    # Load Model:
    model = ConvNet2L().to(device)
    model.load_state_dict(torch.load(MODEL_STORE_PATH+"conv_net_model_2conv_15.pt"))


    # Evaluate Model:
    model.eval()
    test_model(model, test_loader, device)