from train_model import *
import torch
from torch import nn
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms


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


def PlotPrediction(img, pred):
    fig, axs = plt.subplots(1, 2)

    alphab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    frequencies = pred.flatten().cpu().detach().numpy()

    pos = np.arange(len(alphab))
    width = 1.0     # gives histogram aspect to the bar diagram

    axs[0].imshow(img.cpu().view(28, 28))
    # axs[1] = plt.axes()
    axs[1].set_xticks(pos)
    axs[1].set_xticklabels(alphab)
    axs[1].bar(pos, frequencies, width, color='r')
    fig.tight_layout()
    plt.show()
    plt.close('all')


class ImageCapture:
    def __init__(self):
        self.drawing = False
        self.pt1_x = None
        self.pt1_y = None


        self.img = np.zeros((28,28,1), np.float)
        cv2.namedWindow('test draw', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('test draw',self.line_drawing)

        while(1):
            cv2.imshow('test draw', self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

        kernel = np.ones((2, 2),np.float32)/4
        self.img = cv2.filter2D(self.img,-1,kernel)
        

    def line_drawing(self, event, x, y, flags, param):

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

        return self.img

    def get_image(self):
        return self.img


if __name__ == "__main__":
    DATA_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/data/'
    MODEL_STORE_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/model/'

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_STORE_PATH+"conv_net_model_2conv_25.pt"))
    model.eval()




    ### LOAD IMAGE FROM FILE:
    # img = LoadImage(
    #     "/home/aldi/workspace/projects/mnist_cnn/src/data/sample_nums/num_0.png", device)


    ### GET IMAGE FROM MAUSE:
    capture = ImageCapture()
    img = capture.get_image()
    img = pre_process_image(img, device)


    # View Image:
    plt.imshow(img.cpu().view(28, 28))
    plt.show()
    print(img)


    # Evaluate the image:
    img = img.to(device)
    pred = model(img)
    indx = torch.argmax(pred.data, dim=1)
    print(indx, pred)

    PlotPrediction(img, pred)
