from train_model import *
import torch
from torch import nn
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms
from models import *


def LoadImage(file, device):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # print("The max is: ", max_val)
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    img = trans(img).float()
    img = img.view(1, 1, 28, 28)
    return img

def pre_process_image(img, device):
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    img = trans(img).float()
    img = img.view(1, 1, 28, 28)
    return img


def PlotPrediction(img, pred3L):
    fig, axs = plt.subplots(1, 2)

    alphab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    frequencies3L = pred3L.flatten().cpu().detach().numpy()


    pos = np.arange(len(alphab))
    width = 1.0     

    axs[0].imshow(img.cpu().view(28, 28))
    axs[0].title.set_text('Input Image')

    axs[1].set_xticks(pos)
    axs[1].set_xticklabels(alphab)
    axs[1].bar(pos, frequencies3L, width, color='r')
    axs[1].title.set_text('Network Prediction')


    fig.tight_layout()
    plt.show()
    plt.close('all')


class ImageCapture:
    def __init__(self):
        self.drawing = False
        self.pt1_x = None
        self.pt1_y = None


        # self.img = np.zeros((28,28,1), np.float)
        self.img = np.zeros((500,500,1), np.float)
        # self.img = cv2.rectangle(self.img, (50, 50), (250, 250), 50, (255, 255, 255), 5)
        cv2.namedWindow('Draw a number', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Draw a number',self.line_drawing)

        while(1):
            cv2.imshow('Draw a number', self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

        self.CenterImage()
        self.Normalize()

        

    def line_drawing(self, event, x, y, flags, param):

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

    def get_image(self):
        return self.img

    def CenterImage(self):
        self.img = self.img[:,:,0]
        img_cp = np.uint8(self.img)

        height = img_cp.shape[0]
        width = img_cp.shape[1]

        x, y, w, h = cv2.boundingRect(img_cp)

        buff = 50
        self.img = img_cp[y-buff:y+h+buff, x-buff:x+w+buff]

    def Normalize(self):
        self.img = cv2.resize(self.img, (28, 28), interpolation=cv2.INTER_AREA)
        # kernel = np.ones((2, 2),np.float32)/4
        # self.img = cv2.filter2D(self.img,-1,kernel)
        max_val = np.amax(self.img)  
        self.img = self.img / max_val
        # self.img[self.img > 0.6] = 1

        # self.img = 1 - self.img
        return self.img


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (20,10)


    DATA_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/data/'
    MODEL_STORE_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/model/'

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model3L = ConvNet3L().to(device)
    model3L.load_state_dict(torch.load(MODEL_STORE_PATH+"conv_net_model_3conv_15.pt"))
    model3L.eval()


    ### LOAD IMAGE FROM FILE:
    # img = LoadImage(
    #     "/home/aldi/workspace/projects/mnist_cnn/src/data/sample_nums/num_0.png", device)

    ### GET IMAGE FROM MAUSE:
    while(1):
        capture = ImageCapture()
        img = capture.get_image()
        img = pre_process_image(img, device)


        # Evaluate the image:
        img = img.to(device)
        pred3L = model3L(img)
        indx3L = torch.argmax(pred3L.data, dim=1)

        print("Model Prediction ", indx3L)

        PlotPrediction(img, pred3L)
