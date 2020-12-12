import cv2
import numpy as np 
import matplotlib.pyplot as plt
import torch
from train_model import *

def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=40)
            pt1_x,pt1_y=x,y
            
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=40)        

    

if __name__ == "__main__":
    model = GetModelCNNClass()
    model.load_state_dict(torch.load("/home/aldi/workspace/mnist_cnn/src/model/model_cnn"))
    # model.load_state_dict(torch.load("/home/aldi/workspace/mnist_cnn/src/model/model_mlp"))
    model.eval()

    drawing = False # true if mouse is pressed
    pt1_x , pt1_y = None , None
    img = np.zeros((512,512), np.float)
    cv2.namedWindow('test draw')
    cv2.setMouseCallback('test draw',line_drawing)


    while(1):
        cv2.imshow('test draw',img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    img = 255-img


    resized = cv2.resize(img,(28,28), interpolation = cv2.INTER_AREA) 

    x_test = torch.from_numpy(resized)
    pred = model(x_test.float())
    out = torch.argmax(pred, dim=1)
    print("Prediction: ", out, pred)




    plt.imshow(x_test,  cmap="gray")
    plt.show()


