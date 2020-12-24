import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    capture = ImageCapture()
    img = capture.get_image()

    kernel = np.ones((2, 2),np.float32)/4
    img = cv2.filter2D(img,-1,kernel)

    plt.imshow(img)
    plt.show()
