import cv2
import numpy as np 
import matplotlib.pyplot as plt

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)        


img = np.zeros((28,28,1), np.float)
cv2.namedWindow('test draw', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('test draw',line_drawing)

while(1):
    cv2.imshow('test draw',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()

print("End of life")

kernel = np.ones((2,2),np.float32)/4
img = cv2.filter2D(img,-1,kernel)

cv2.imwrite("test7.png", img) 

plt.imshow(img)
plt.show()
