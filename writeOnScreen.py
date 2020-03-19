import imutils
import cv2
import numpy as np
import random as rnd

def get_mask(image):
    #skin colour range
    lower_bound = np.array([0,133,77], dtype=np.uint8)
    upper_bound = np.array([235,173,127], dtype=np.uint8)

    mask = cv2.inRange(image, lower_bound, upper_bound)
    #removing the noise 
    kernelOpen = np.ones((4, 4))
    kernelClose = np.ones((10, 10))

    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    final_mask = maskClose
    return final_mask

def get_contours(img):
    contours= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    myCam = cv2.VideoCapture(0)
    frameBlank = np.zeros((500, 500, 3), dtype=np.uint8)

    while True:
        ret, frameCap = myCam.read()
        frameCap = cv2.flip(frameCap, 1)
        frameCap = cv2.resize(frameCap, (500, 500))

        ycr_image = cv2.cvtColor(frameCap, cv2.COLOR_BGR2YCR_CB)
        mask = get_mask(ycr_image)
        contours = get_contours(mask)

        cn = imutils.grab_contours(contours)
        maxCn = max(cn, key=cv2.contourArea)
        
        # determine the most extreme points along the contour
        # extLeft = tuple(c[c[:, :, 0].argmin()][0])
        # extRight = tuple(c[c[:, :, 0].argmax()][0])

        extTop = tuple(maxCn[maxCn[:, :, 1].argmin()][0])

        color = (1, 45, 225)
        cv2.drawContours(frameCap, [maxCn], -1, color, 2)

        # cv2.circle(frameCap, extLeft, 8, (0, 0, 255), -1)
        # cv2.circle(frameCap, extRight, 8, (0, 255, 0), -1)
        cv2.circle(frameCap, extTop, 5, (255, 0, 0), -1)
        cv2.circle(frameBlank, extTop, 8, (255, 255, 255), -1)
        
        frameMerged = cv2.add(frameCap, frameBlank)
        cv2.imshow("Write", frameMerged)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    myCam.release()


if __name__ == "__main__":
    main()
