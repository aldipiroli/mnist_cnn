# from train_.model import *
import torch
from torch import nn
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms
from models import *
import numpy as np
import matplotlib.pyplot as plt


def LoadImage(file, device):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    img = trans(img).float()
    img = img.view(1, 1, 28, 28)
    return img


class RecognizeNumber:
    def __init__(self, model_ID_):
        self.DATA_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/data/'
        self.model_STORE_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/self.model/'

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model_ID = model_ID_
        self.model = None

        ### Load the model:
        self.LoadModel()

        self.drawing = False
        self.pt1_x = None
        self.pt1_y = None

        self.img = np.zeros((500, 500, 1), np.float)
        cv2.namedWindow('test draw', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('test draw', self.LideDrawing)

        while(1):
            cv2.imshow('test draw', self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

        self.RecognizeImage()

        # self.PlotPrediction()


    def LideDrawing(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1_x, self.pt1_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.line(self.img, (self.pt1_x, self.pt1_y),
                         (x, y), color=(255, 255, 255), thickness=20)
                self.pt1_x, self.pt1_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.img, (self.pt1_x, self.pt1_y),
                     (x, y), color=(255, 255, 255), thickness=20)

        return self.img

    def RecognizeImage(self):
        self.CenterImage()
        print("CenterImage")

        self.Normalize()
        print("Normalize")

        img = self.ProcessImage()
        print("ProcessImage")

        self.MakePrediction(img)
        print("MakePrediction")



    def CenterImage(self):
        self.img = self.img[:, :, 0]
        img_cp = np.uint8(self.img)

        height = img_cp.shape[0]
        width = img_cp.shape[1]

        x, y, w, h = cv2.boundingRect(img_cp)

        buff = 50
        self.img = img_cp[y-buff:y+h+buff, x-buff:x+w+buff]

    def Normalize(self):
        self.img = cv2.resize(self.img, (28, 28), interpolation=cv2.INTER_AREA)
        self.img = self.img / np.amax(self.img)
        return self.img

    def ProcessImage(self, img, device):
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        img = trans(img).float()
        img = img.view(1, 1, 28, 28)
        return img

    def PlotPrediction(self):
        fig, axs = plt.subplots(1, 2)

        alphab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        frequencies = self.pred.flatten().cpu().detach().numpy()

        pos = np.arange(len(alphab))

        axs[0].imshow(self.img.cpu().view(28, 28))

        axs[1].set_xticks(pos)
        axs[1].set_xticklabels(alphab)
        axs[1].bar(pos, frequencies, 2, color='r')

        # axs[2].set_xticks(pos)
        # axs[2].set_xticklabels(alphab)
        # axs[2].bar(pos, frequencies2L, 2, color='r')

        # fig.suptitle('test title', fontsize=12)

        fig.tight_layout()
        plt.show()
        plt.close('all')

    def LoadModel(self):
        if(self.model_ID != 2 or self.model_ID != 3):
            print("ERROR in the self.model Choiche!")
            return

        if(self.model_ID == 2):
            self.model = ConvNet2L().to(device)
            self.model.load_state_dict(torch.load(
                self.model_STORE_PATH+"conv_net_self.model_2conv_15.pt"))

        if(self.model_ID == 3):
            self.model = ConvNet3L().to(device)
            self.model.load_state_dict(torch.load(self.model_STORE_PATH+"conv_net_self.model_3conv_15.pt"))
        
        
        self.model.eval()

    def MakePrediction(self, img_):
        img = img_.to(device)
        self.pred = self.model(img)
        self.indx = torch.argmax(self.pred.data, dim=1)
        print("Prediction", self.indx)

    def GetImage(self):
        return self.img


if __name__ == "__main__":

    r = RecognizeNumber(2)