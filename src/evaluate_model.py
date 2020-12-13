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

from train_model import *


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


def LoadImage(file):
    img = cv2.imread(file, 0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = np.floor(img / 255)
    img = torch.from_numpy(img)
    img = img.view(1, 784)
    img = img.float()
    return img


class ImageHandler:
    def __init__(self):
        # super().__init__()
        self.drawing = False
        self.pt1_x = None
        self.pt1_y = None

        self.img = np.zeros((28, 28, 1), np.float)
        cv2.namedWindow('test draw', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('test draw', self.line_drawing)

        while(1):
            cv2.imshow('test draw', self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            # self.EvaluateImage()

        cv2.destroyAllWindows()
        
    def line_drawing(self,event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1_x, self.pt1_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.line(self.img, (self.pt1_x, self.pt1_y),
                         (x, y), color=(255, 255, 255), thickness=2)
                self.pt1_x, self.pt1_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.img, (self.pt1_x, self.pt1_y),
                     (x, y), color=(255, 255, 255), thickness=2)

    
    def ProcessImage(self):
        self.img = cv2.resize(self.img, (28, 28), interpolation=cv2.INTER_AREA)
        self.img = np.floor(self.img / 255)
        self.img = torch.from_numpy(self.img)
        self.img = self.img.view(1, 784)
        self.img = self.img.float()
        return self.img


    def EvaluateImage(self):
        img = self.ProcessImage()
        pred = model(img)
        indx = torch.argmax(pred, dim=1)
        plt.imshow(img.view(28, 28))
        print("Prediction: ", indx)



if __name__ == "__main__":
    model = Mnist_CNN()
    model.load_state_dict(torch.load(
        "/home/aldi/workspace/mnist_cnn/src/model/model_cnn.pt"))
    model.eval()

    # file1 = "/home/aldi/workspace/mnist_cnn/src/data/mnist/sample_mnist/tensor([3]).png"
    # file2 = "/home/aldi/workspace/mnist_cnn/src/data/mnist/sample_nums/num_8.png"
    # img = LoadImage(file2)
    handler = ImageHandler()
    img = handler.ProcessImage()


    pred = model(img)
    indx = torch.argmax(pred, dim=1)
    plt.imshow(img.view(28, 28))
    print("Prediction: ", indx)
    plt.show()


    # train_dl, valid_dl, n, c = LoadData(1)

    # for xb, yb in valid_dl:
    #     # plt.imshow(xb.view(28, 28))
    #     indx = torch.argmax(model(xb), dim=1)
    #     print("Pred: ", indx, ", Truth:", yb)
    #     # cv2.imwrite("/home/aldi/workspace/mnist_cnn/src/data/mnist/sample_mnist/"+str(yb.flatten())+".png", xb.view(28, 28).numpy())
    #     plt.show()
    #     input()
