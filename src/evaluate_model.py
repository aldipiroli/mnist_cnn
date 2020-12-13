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

import cv2


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
    valid_dl = DataLoader(valid_ds, batch_size=bs)

    return train_dl, valid_dl, n, c


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


def LoadImage(file):
    img = cv2.imread(file, 0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # img = np.floor( img / 255)
    img = img / 255 
    img = torch.from_numpy(img)
    img = img.view(1, 784)
    img = img.float()
    print(max(img))
    # print(img.size())

    # plt.imshow(img)
    # plt.show()
    return img


if __name__ == "__main__":
    model = Mnist_CNN()
    model.load_state_dict(torch.load(
        "/home/aldi/workspace/mnist_cnn/src/model/model_cnn.pt"))
    model.eval()

    file1 = "/home/aldi/workspace/mnist_cnn/src/data/mnist/sample_mnist/tensor([3]).png"
    file2 = "/home/aldi/workspace/mnist_cnn/src/data/mnist/sample_nums/num_8.png"
    img = LoadImage(file2)

    pred = model(img)
    indx = torch.argmax(pred, dim=1)
    plt.imshow(img.view(28,28))
    print("Prediction: ",indx)
    plt.show()

    # train_dl, valid_dl, n, c = LoadData(1)

    # for xb, yb in valid_dl:
    #     # plt.imshow(xb.view(28, 28))
    #     indx = torch.argmax(model(xb), dim=1)
    #     print("Pred: ", indx, ", Truth:", yb)
    #     # cv2.imwrite("/home/aldi/workspace/mnist_cnn/src/data/mnist/sample_mnist/"+str(yb.flatten())+".png", xb.view(28, 28).numpy()) 
    #     plt.show()
    #     input()
