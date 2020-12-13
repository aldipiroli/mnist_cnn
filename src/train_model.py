import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import math
import random

from pathlib import Path
import requests
import pickle
import gzip
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch import optim


def LoadData(bs):
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid),
            _) = pickle.load(f, encoding="latin-1")

    # Convert np -> torch tensors
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )

    n, c = x_train.shape

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    return train_dl, valid_dl, n, c


class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
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
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
    #     self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
    #     self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    # def forward(self, xb):
    #     xb = xb.view(-1, 1, 28, 28)
    #     xb = F.relu(self.conv1(xb))
    #     xb = F.relu(self.conv2(xb))
    #     xb = F.relu(self.conv3(xb))
    #     xb = F.avg_pool2d(xb, 4)
    #     return xb.view(-1, xb.size(1))



if __name__ == "__main__":
    bs = 64
    epochs = 5
    lr = 0.1

    train_dl, valid_dl, n, c = LoadData(bs)
    loss_func = F.cross_entropy

    model = Mnist_CNN()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    for epoch in range(epochs):
        model.train()
        for count, (xb, yb) in enumerate(train_dl):
            if count % 100 == 0:
                print("Epoch: ", epoch, " #: ", count)

            # plt.imshow(xb.view((28,28)),  cmap="gray")
            # plt.show()
            # input()
            # print(xb.size())

            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

        print(epoch, valid_loss / len(valid_dl))

    # Save Model:
    torch.save(model.state_dict(), "/home/aldi/workspace/mnist_cnn/src/model/model_cnn.pt")
    # torch.save(model, "/home/aldi/workspace/mnist_cnn/src/model/model")


    # Model class must be defined somewhere
    # model = torch.load(PATH)
    # model.eval()