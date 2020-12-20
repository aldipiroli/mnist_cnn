from train_model import *
import torch
from torch import nn
import cv2
from torch.autograd import Variable



def LoadImage(file, device):
    img = cv2.imread(file, 0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # img = np.floor(img / 255)
    img = img / 255

    img = torch.from_numpy(img)
    img = img.view(1, 1, 28, 28)
    img = img.float()
    return img


if __name__ == "__main__":
    DATA_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/data/'
    MODEL_STORE_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/model/'

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_STORE_PATH+"conv_net_model.pt"))
    model.eval()

    img = LoadImage(
        DATA_PATH + "sample_nums/num_9.png", device)
    img = img.to(device)

    print(img.shape)
    pred = model(img)
    indx = torch.argmax(pred.data, dim=1)
    plt.imshow(img.cpu().view(28, 28))
    print("Prediction: ", indx.item(), "",  pred)
    plt.show()
