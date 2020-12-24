from train_model import *
import torch
from torch import nn
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms


def LoadImage(file, device):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    img = trans(img)
    img = img.view(1, 1, 28, 28)
    # print(img)
    return img


def PlotPrediction(img, pred):
    fig, axs = plt.subplots(1, 2)

    alphab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    frequencies = pred.flatten().cpu().detach().numpy()

    pos = np.arange(len(alphab))
    width = 1.0     # gives histogram aspect to the bar diagram

    axs[0].imshow(img.cpu().view(28, 28))
    # axs[1] = plt.axes()
    axs[1].set_xticks(pos )
    axs[1].set_xticklabels(alphab)
    axs[1].bar(pos, frequencies, width, color='r')
    fig.tight_layout()
    plt.show()
    plt.close('all')


def load_dataset(DATA_PATH, batch_size):
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # MNIST dataset
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH, train=False, transform=trans)

    # Data loader
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)
    
    return test_loader

if __name__ == "__main__":
    DATA_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/data/'
    MODEL_STORE_PATH = '/home/aldi/workspace/projects/mnist_cnn/src/model/'

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_STORE_PATH+"conv_net_model.pt"))
    model.eval()


    # test_loader = load_dataset(DATA_PATH, batch_size=1)

    
    img = LoadImage("/home/aldi/workspace/projects/mnist_cnn/src/data/mnist_sample/test7.png", device)

    print(img.shape)
    # img = LoadImage(
    #     DATA_PATH + "mnist_sample/6.png", device)
    img = img.to(device)

    pred = model(img)
    indx = torch.argmax(pred.data, dim=1)  
    

    print(indx, pred)
    cv2.imwrite("test7.png", img.cpu().view(28,28).numpy()) 

    # print(classes)
    # PlotPrediction(img, pred)
